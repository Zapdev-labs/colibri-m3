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

SKIP_RE = re.compile(r"(indexer|eh_proj|enorm|hnorm|shared_head|mtp|nextn|vision|visual)")
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
SHARED_RE = re.compile(r"\.mlp\.shared_mlp\.")


def layer_idx(name: str) -> int:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def quant_int8(w: np.ndarray):
    o, i = w.shape
    q = np.empty((o, i), dtype=np.int8)
    s = np.empty(o, dtype=np.float32)
    for r in range(o):
        amax = float(np.max(np.abs(w[r])))
        scale = amax / 127.0 if amax > 1e-8 else 1e-8
        s[r] = scale
        q[r] = np.clip(np.rint(w[r] / scale), -128, 127).astype(np.int8)
    return q, s


def quant_int4(w: np.ndarray):
    o, i = w.shape
    rb = (i + 1) // 2
    q4 = np.zeros((o, rb), dtype=np.uint8)
    s = np.empty(o, dtype=np.float32)
    for r in range(o):
        amax = float(np.max(np.abs(w[r])))
        scale = amax / 7.0 if amax > 1e-8 else 1e-8
        s[r] = scale
        q = np.clip(np.rint(w[r] / scale), -8, 7).astype(np.int32)
        packed = np.zeros(rb, dtype=np.uint8)
        for c in range(0, i, 2):
            lo = int(q[c]) + 8
            hi = (int(q[c + 1]) + 8) if c + 1 < i else 8
            packed[c >> 1] = (lo & 0xF) | ((hi & 0xF) << 4)
        q4[r] = packed
    return q4, s


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
    eos = raw.get("eos_token_id", tc.get("eos_token_id"))
    if isinstance(eos, list):
        eos = eos[0] if eos else 200020
    theta = tc.get("rope_theta", rope.get("rope_theta", 5000000.0))
    return {
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


def should_skip(name: str, n_layers: int) -> bool:
    if SKIP_RE.search(name):
        return True
    if name.startswith("vision_model.") or name.startswith("model.visual."):
        return True
    li = layer_idx(name)
    return li >= n_layers if li >= 0 else False


def classify(name: str) -> str:
    if name.endswith("_scale") or name.endswith("_scale_inv"):
        return "skip"
    if name.endswith("e_score_correction_bias") or name.endswith("expert_bias"):
        return "f32"
    if name.endswith("mlp.gate.weight") or name.endswith("mlp.router.gate.weight"):
        return "f32"
    if name.endswith("norm.weight") or name == "model.norm.weight":
        return "f32"
    if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
        return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "io"
    if ".mlp.experts." in name and name.endswith(".weight"):
        return "expert"
    if name.endswith(".weight"):
        return "dense"
    return "f32"


def rename(name: str) -> str:
    return SHARED_RE.sub(".mlp.shared_experts.", name)


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


def convert_local(indir: str, outdir: str, n_layers: int, ebits: int, io_bits: int, dbits: int):
    from safetensors.numpy import save_file

    os.makedirs(outdir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors in {indir}")
    for i, sp in enumerate(shards):
        out_path = os.path.join(outdir, f"out-{i:05d}.safetensors")
        if os.path.isfile(out_path):
            print(f"[{i+1}/{len(shards)}] skip existing {os.path.basename(out_path)}")
            continue
        print(f"[{i+1}/{len(shards)}] {os.path.basename(sp)}")
        out = convert_file(sp, n_layers, ebits, io_bits, dbits)
        save_file(out, out_path)
        print(f"    -> {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)")
    write_sidecar(indir, outdir)
    print(f"done: {len(shards)} shards -> {outdir}")


def convert_repo(repo: str, outdir: str, n_layers: int, ebits: int, io_bits: int, dbits: int):
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.numpy import save_file

    os.makedirs(outdir, exist_ok=True)
    files = [f for f in list_repo_files(repo) if f.endswith(".safetensors") and f.startswith("model-")]
    files.sort()
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
        out_path = os.path.join(outdir, f"out-{i:05d}.safetensors")
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
    a = ap.parse_args()
    dbits = a.dbits if a.dbits is not None else a.ebits
    if a.indir:
        convert_local(a.indir, a.outdir, a.n_layers, a.ebits, a.io_bits, dbits)
    elif a.repo:
        convert_repo(a.repo, a.outdir, a.n_layers, a.ebits, a.io_bits, dbits)
    else:
        sys.exit("pass --repo or --indir")


if __name__ == "__main__":
    main()
