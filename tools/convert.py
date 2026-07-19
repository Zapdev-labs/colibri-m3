#!/usr/bin/env python3
"""Standalone MiniMax-M3 → int4 streaming container converter. No upstream imports."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

# Tensors to skip entirely (vision/mtp/nextn heads not used by the M3 text engine).
# NOTE: `indexer` matches the literal substring "indexer" (a separate HF module),
# NOT "index_q_proj" / "index_k_proj" — those MSA indexer tensors are kept and
# classified as attn/f32 below.
SKIP_RE = re.compile(r"(indexer|eh_proj|enorm|hnorm|shared_head|mtp|nextn|vision|visual)")
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
SHARED_RE = re.compile(r"\.mlp\.shared_mlp\.")

# MiniMax-M3 EOS token id (from generation_config.json). Used as a fallback when
# the flattened config.json reports eos_token_id: null.
M3_EOS_TOKEN_ID = 200020


def layer_idx(name: str) -> int:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def _normalize(name: str) -> str:
    """Normalize a (possibly raw HF) tensor name to the engine's canonical form.

    Three rewrites (the audit's converter bugs 1 + 2 + shared_mlp):
      1. Strip the ``language_model.`` prefix.
      2. Rewrite HF's ``block_sparse_moe`` module component → ``mlp``.
      3. Rewrite ``shared_mlp`` → ``shared_experts`` (older HF naming variant).
    A side-effect of (2): ``block_sparse_moe.e_score_correction_bias`` becomes
    ``mlp.e_score_correction_bias``, but the engine requests it under
    ``mlp.gate.e_score_correction_bias`` — so insert ``.gate`` there too.

    This function is idempotent: passing an already-canonical name returns it
    unchanged. Both ``rename()`` and ``classify()`` call it so ``classify()``
    is prefix-agnostic and works on raw HF names directly (per VAL-FOUND-004).
    """
    # (1) Strip language_model. prefix (idempotent: no-op if absent).
    name = name.removeprefix("language_model.")
    # (2) Rewrite block_sparse_moe -> mlp (idempotent: no-op if absent).
    name = name.replace("block_sparse_moe", "mlp")
    # e_score_correction_bias / expert_bias live directly under block_sparse_moe
    # in HF's checkpoint, but the engine requests them under ``mlp.gate.*``.
    # After (2) they became ``mlp.e_score_correction_bias`` / ``mlp.expert_bias``;
    # insert the missing ``.gate`` component.
    name = name.replace(".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
    name = name.replace(".mlp.expert_bias", ".mlp.gate.expert_bias")
    # (3) Rewrite shared_mlp -> shared_experts (idempotent: no-op if absent).
    name = SHARED_RE.sub(".mlp.shared_experts.", name)
    # (4) Rewrite HF per-expert weight naming w1/w2/w3 -> gate_proj/down_proj/up_proj.
    # HF MiniMax-M3 stores routed experts as experts.E.{w1,w2,w3}.weight where
    # w1=gate (hidden→inter), w3=up (hidden→inter), w2=down (inter→hidden).
    # The engine requests experts.E.{gate_proj,up_proj,down_proj}.weight, so:
    #   w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj.
    name = re.sub(r"\.experts\.(\d+)\.w1\.weight", r".experts.\1.gate_proj.weight", name)
    name = re.sub(r"\.experts\.(\d+)\.w3\.weight", r".experts.\1.up_proj.weight", name)
    name = re.sub(r"\.experts\.(\d+)\.w2\.weight", r".experts.\1.down_proj.weight", name)
    return name


def quant_int8(w: np.ndarray):
    """Per-row symmetric int8 quantization. Vectorized with numpy broadcasting.

    Bit-exact with the original per-row Python loop: for each row r, computes
    ``amax = max(abs(w[r]))`` (as a Python float, which numpy computes in the
    array's native dtype then upcasts to float64 for ``float(...)``), ``scale =
    amax/127`` (or 1e-8 floor), then ``q[r] = clip(rint(w[r]/scale), -128, 127)``
    as int8. Critically, ``w[r] / scale`` is performed in float32 when ``w`` is
    float32 (numpy's scalar/array promotion): the per-row ``scale`` python float
    preserves float32 precision in the division. We reproduce that by keeping
    the division in float32: cast ``scale`` back to float32 and divide.
    """
    o, i = w.shape
    # Match the original loop: amax computed in float32 (numpy's max on the
    # float32 row), then float() upcasts to python float64.
    amax = np.max(np.abs(w), axis=1).astype(np.float64)
    scale64 = np.where(amax > 1e-8, amax / 127.0, 1e-8)
    s = scale64.astype(np.float32)
    # The original loop computes w[r] / scale where w[r] is float32 and scale is
    # python float (float64). Numpy promotes the result to float32 because the
    # array operand w[r] is float32 and python-float / np.float32 stays float32
    # when the array is the dividend. Reproduce by dividing in float32.
    scale32 = scale64.astype(np.float32)
    q = np.clip(np.rint(w.astype(np.float32) / scale32[:, None]), -128, 127).astype(np.int8)
    return q, s


def quant_int4(w: np.ndarray):
    """Per-row symmetric int4 quantization with two-nibbles-per-byte packing.

    Vectorized with numpy broadcasting; bit-exact with the original per-row loop.
    Each row r: ``amax = max(abs(w[r]))`` (float64 via python float), ``scale =
    amax/7`` (floor 1e-8), ``q = clip(rint(w[r]/scale), -8, 7)``. Then each int
    is shifted to unsigned (``q+8`` -> 0..15) and packed two-per-byte (low
    nibble first). If the row width is odd, the trailing high nibble is 8
    (neutral). The division ``w[r]/scale`` runs in float32 to match the loop's
    numpy scalar/array promotion exactly.
    """
    o, i = w.shape
    rb = (i + 1) // 2
    amax = np.max(np.abs(w), axis=1).astype(np.float64)
    scale64 = np.where(amax > 1e-8, amax / 7.0, 1e-8)
    s = scale64.astype(np.float32)
    # Divide in float32 to match the original loop's promotion behavior.
    scale32 = scale64.astype(np.float32)
    q = np.clip(np.rint(w.astype(np.float32) / scale32[:, None]), -8, 7).astype(np.int32) + 8
    if i % 2 == 1:
        pad = np.full((o, 1), 8, dtype=np.int32)
        q = np.concatenate([q, pad], axis=1)
    q2 = q.reshape(o, rb, 2)
    lo = (q2[:, :, 0] & 0xF).astype(np.uint8)
    hi = (q2[:, :, 1] & 0xF).astype(np.uint8)
    packed = lo | (hi << 4)
    return packed, s


def flatten_config(raw: dict) -> dict:
    tc = raw.get("text_config", raw)
    moe_freq = tc.get("moe_layer_freq")
    mlp_types = tc.get("mlp_layer_types")
    if moe_freq is not None:
        first = 0
        for v in moe_freq:
            if int(v) == 0:
                first += 1
            else:
                break
    elif mlp_types:
        first = 0
        for t in mlp_types:
            if t == "dense":
                first += 1
            else:
                break
    else:
        first = 0
    rope = tc.get("rope_parameters") or {}
    # eos_token_id: HF ships null in config.json for this checkpoint; fall back
    # to the M3 generation_config.json value (200020). Fixes the audit's
    # "eos_token_id null -> 0 at runtime" bug (VAL-FOUND-005 / VAL-CORR-017).
    eos = raw.get("eos_token_id", tc.get("eos_token_id"))
    if isinstance(eos, list):
        eos = eos[0] if eos else M3_EOS_TOKEN_ID
    if eos is None:
        eos = M3_EOS_TOKEN_ID
    theta = tc.get("rope_theta", rope.get("rope_theta", 5000000.0))
    cfg = {
        "model_type": "minimax_m3",
        "hidden_size": tc["hidden_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc.get("head_dim", 128),
        "vocab_size": tc["vocab_size"],
        "num_experts": tc.get("num_local_experts", tc.get("num_experts", 128)),
        "num_experts_per_tok": tc.get("num_experts_per_tok", 4),
        "moe_intermediate_size": tc.get("intermediate_size", 3072),
        "intermediate_size": tc.get("dense_intermediate_size", 12288),
        "dense_intermediate_size": tc.get("dense_intermediate_size", 12288),
        "first_k_dense_replace": first,
        "num_shared_experts": tc.get("n_shared_experts", tc.get("num_shared_experts", 1)),
        "router_scaling_factor": tc.get("routed_scaling_factor", tc.get("router_scaling_factor", 2.0)),
        "routed_scaling_factor": tc.get("routed_scaling_factor", 2.0),
        "route_norm": True,
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "rope_theta": theta,
        "rope_parameters": {"rope_theta": theta},
        "rotary_dim": tc.get("rotary_dim", 64),
        "swiglu_alpha": tc.get("swiglu_alpha", 1.702),
        "swiglu_limit": tc.get("swiglu_limit", 7.0),
        "use_gemma_norm": tc.get("use_gemma_norm", True),
        "eos_token_id": eos,
        "max_position_embeddings": tc.get("max_position_embeddings", 1048576),
    }
    # Preserve sparse_attention_config if present (needed by the MSA port).
    sac = tc.get("sparse_attention_config") or raw.get("sparse_attention_config")
    if sac:
        cfg["sparse_attention_config"] = sac
    return cfg


def should_skip(name: str, n_layers: int) -> bool:
    if SKIP_RE.search(name):
        return True
    if name.startswith("vision_model.") or name.startswith("model.visual."):
        return True
    li = layer_idx(name)
    return li >= n_layers if li >= 0 else False


def classify(name: str) -> str:
    """Classify a tensor name into a quantization class.

    Prefix-agnostic: operates on the canonical (normalized) name, so it returns
    the same answer whether given a raw HF name (``language_model.model...``) or
    a stripped engine name (``model...``). Per VAL-FOUND-004 the policy is:

      embed_tokens / lm_head          -> io     (int8,  io_bits=8)
      *.mlp.gate.weight (router)      -> f32
      *.mlp.gate.e_score_correction_bias -> f32
      *.mlp.experts.*  (routed)       -> expert (int4, ebits=4)
      *.mlp.shared_experts.*          -> expert (int4, ebits=4)
      all *norm* (layernorm/final/qk/index norms) -> f32
      *.self_attn.{q,k,v,o}_proj      -> attn   (int4, dbits=4)
      *.self_attn.index_{q,k}_proj    -> attn   (int4, dbits=4)
      dense mlp.{gate,up,down}_proj   -> dense  (int4, dbits=4)
    """
    n = _normalize(name)
    if n.endswith("_scale") or n.endswith("_scale_inv") or n.endswith(".qs"):
        return "skip"
    # Router bias / expert bias -> f32 (checked before router gate so the
    # ``.gate.e_score_correction_bias`` suffix lands here, not in the gate branch).
    if "e_score_correction_bias" in n or n.endswith("expert_bias"):
        return "f32"
    # MoE router gate weight -> f32.
    if ".mlp.gate.weight" in n:
        return "f32"
    # All norms (input_layernorm, post_attention_layernorm, model.norm,
    # per-head q_norm/k_norm, index_q_norm/index_k_norm) -> f32.
    if "norm.weight" in n or n == "model.norm.weight":
        return "f32"
    # Embeddings and lm_head -> int8 (io).
    if n == "model.embed_tokens.weight" or n == "lm_head.weight":
        return "io"
    # Routed experts -> int4 (expert).
    if ".mlp.experts." in n and n.endswith(".weight"):
        return "expert"
    # Shared experts -> int4 (expert).
    if ".mlp.shared_experts." in n and n.endswith(".weight"):
        return "expert"
    # Attention projections (q/k/v/o + index_q_proj/index_k_proj) -> int4 (attn).
    if ".self_attn." in n and n.endswith("_proj.weight"):
        return "attn"
    # Fallback: any other weight tensor -> dense (int4 when dbits=4).
    if n.endswith(".weight"):
        return "dense"
    return "f32"


def rename(name: str) -> str:
    """Rewrite a raw HF tensor name to the engine's canonical form.

    See ``_normalize`` for the full set of rewrites. This is a thin wrapper
    kept as a public entry point so callers (and tests) can exercise the
    renaming in isolation from classify().
    """
    return _normalize(name)


def expand(name: str, w: np.ndarray):
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    if base.endswith(".mlp.experts.gate_up_proj") and w.ndim == 3:
        e, two_i, h = w.shape
        i = two_i // 2
        for ei in range(e):
            prefix = f"model.layers.{layer_idx(name)}.mlp.experts.{ei}."
            yield prefix + "gate_proj.weight", w[ei, :i]
            yield prefix + "up_proj.weight", w[ei, i:]
        return
    if base.endswith(".mlp.experts.down_proj") and w.ndim == 3:
        e, h, i = w.shape
        for ei in range(e):
            yield f"model.layers.{layer_idx(name)}.mlp.experts.{ei}.down_proj.weight", w[ei]
        return
    if base.endswith("gate_up_proj") and w.ndim == 2:
        two_i, h = w.shape
        i = two_i // 2
        prefix = base[: -len("gate_up_proj")]
        yield prefix + "gate_proj.weight", w[:i]
        yield prefix + "up_proj.weight", w[i:]
        return
    yield name, w


def put(out: dict, name: str, w: np.ndarray, ebits: int, io_bits: int, dbits: int):
    name = rename(name)
    kind = classify(name)
    if kind == "skip":
        return
    if kind == "f32" or w.ndim != 2:
        out[name] = w.astype(np.float32)
        return
    bits = io_bits if kind == "io" else ebits if kind == "expert" else dbits
    if bits <= 4:
        q, s = quant_int4(w.astype(np.float32))
    else:
        q, s = quant_int8(w.astype(np.float32))
    out[name] = q
    out[name + ".qs"] = s


def convert_file(path: str, n_layers: int, ebits: int, io_bits: int, dbits: int) -> dict:
    from safetensors import safe_open
    import torch

    out: dict = {}
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            if should_skip(name, n_layers):
                continue
            w = f.get_tensor(name).to(torch.float32).numpy()
            for oname, arr in expand(name, w):
                put(out, oname, arr, ebits, io_bits, dbits)
    return out


def write_sidecar(indir: str, outdir: str, raw_cfg: dict | None = None):
    if raw_cfg is None:
        cfg_path = os.path.join(indir, "config.json")
        if os.path.isfile(cfg_path):
            raw_cfg = json.load(open(cfg_path))
    if raw_cfg is not None:
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(flatten_config(raw_cfg), f, indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json", "special_tokens_map.json"):
        src = os.path.join(indir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, outdir)


def convert_local(indir: str, outdir: str, n_layers: int, ebits: int, io_bits: int, dbits: int,
                  max_shards: int | None = None):
    from safetensors.numpy import save_file

    os.makedirs(outdir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors in {indir}")
    total = len(shards)
    if max_shards is not None:
        shards = shards[:max_shards]
    for i, sp in enumerate(shards):
        out_path = os.path.join(outdir, f"out-{i:05d}-of-{total:05d}.safetensors")
        if os.path.isfile(out_path):
            print(f"[{i+1}/{len(shards)}] skip existing {os.path.basename(out_path)}")
            continue
        print(f"[{i+1}/{len(shards)}] {os.path.basename(sp)}")
        out = convert_file(sp, n_layers, ebits, io_bits, dbits)
        save_file(out, out_path)
        print(f"    -> {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)")
    write_sidecar(indir, outdir)
    print(f"done: {len(shards)} shards -> {outdir}")


def convert_repo(repo: str, outdir: str, n_layers: int, ebits: int, io_bits: int, dbits: int,
                 max_shards: int | None = None):
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.numpy import save_file

    os.makedirs(outdir, exist_ok=True)
    files = [f for f in list_repo_files(repo) if f.endswith(".safetensors") and f.startswith("model-")]
    files.sort()
    total = len(files)
    if max_shards is not None:
        files = files[:max_shards]
    meta = tempfile.mkdtemp(prefix="m3meta_", dir=outdir)
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json"):
        try:
            hf_hub_download(repo, fn, local_dir=meta)
        except Exception:
            pass
    raw = json.load(open(os.path.join(meta, "config.json"))) if os.path.isfile(os.path.join(meta, "config.json")) else None
    if raw:
        tc = raw.get("text_config", raw)
        n_layers = int(tc.get("num_hidden_layers", n_layers))
        write_sidecar(meta, outdir, raw)

    for i, fn in enumerate(files):
        out_path = os.path.join(outdir, f"out-{i:05d}-of-{total:05d}.safetensors")
        if os.path.isfile(out_path):
            print(f"[{i+1}/{len(files)}] skip {os.path.basename(out_path)}")
            continue
        print(f"[{i+1}/{len(files)}] downloading {fn}...")
        path = hf_hub_download(repo, fn, local_dir=os.path.join(outdir, "_inflight"))
        try:
            out = convert_file(path, n_layers, ebits, io_bits, dbits)
            save_file(out, out_path)
            print(f"    -> {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    if raw:
        write_sidecar(meta, outdir, raw)
    print(f"done: {len(files)} shards -> {outdir}")


def main():
    ap = argparse.ArgumentParser(description="MiniMax-M3 → colibri-m3 int4 container")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ebits", type=int, default=4)
    ap.add_argument("--io-bits", type=int, default=8)
    ap.add_argument("--dbits", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=60)
    ap.add_argument("--max-shards", type=int, default=None,
                   help="convert only N shards (smoke mode, e.g. --max-shards 1)")
    a = ap.parse_args()
    dbits = a.dbits if a.dbits is not None else a.ebits
    if a.indir:
        convert_local(a.indir, a.outdir, a.n_layers, a.ebits, a.io_bits, dbits, a.max_shards)
    elif a.repo:
        convert_repo(a.repo, a.outdir, a.n_layers, a.ebits, a.io_bits, dbits, a.max_shards)
    else:
        sys.exit("pass --repo or --indir")


if __name__ == "__main__":
    main()
