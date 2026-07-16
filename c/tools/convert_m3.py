"""Convert MiniMaxAI/MiniMax-M3 (BF16 safetensors) -> Colibri int4 streaming container.

Unpacks 3D expert tensors (gate_up_proj / down_proj), flattens text_config for the C engine.

Usage:
  python3 tools/convert_m3.py --indir ./MiniMax-M3 --outdir ./m3_i4 --ebits 4
  python3 tools/convert_m3.py --repo MiniMaxAI/MiniMax-M3 --outdir /data/m3_i4
  python3 tools/convert_m3.py --repo MiniMaxAI/MiniMax-M3 --outdir /data/m3_i4 --n-layers 60 --io-bits 8
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from convert_fp8_to_int4 import quant_int2, quant_int4, quant_int8, free_gb, layer_idx  # noqa: E402

SHARED_RE = re.compile(r"\.mlp\.shared_mlp\.")
SKIP_RE = re.compile(
    r"(indexer|indexers_proj|eh_proj|enorm|hnorm|shared_head|mtp|nextn|vision|visual)"
)


def flatten_config(raw):
    tc = raw.get("text_config", raw)
    mlp_types = tc.get("mlp_layer_types")
    moe_freq = tc.get("moe_layer_freq")
    if mlp_types:
        first_dense = sum(1 for t in mlp_types if t == "dense")
    elif moe_freq:
        first_dense = 0
        for v in moe_freq:
            if v == 0:
                first_dense += 1
            else:
                break
    else:
        first_dense = 0
    rope = tc.get("rope_parameters") or {}
    eos = raw.get("eos_token_id", tc.get("eos_token_id"))
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    return {
        "model_type": "minimax_m3",
        "hidden_size": tc["hidden_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "vocab_size": tc["vocab_size"],
        "num_experts": tc.get("num_local_experts", tc.get("num_experts", 128)),
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "moe_intermediate_size": tc.get("intermediate_size", 3072),
        "intermediate_size": tc.get("dense_intermediate_size", 12288),
        "first_k_dense_replace": first_dense,
        "num_shared_experts": tc.get("n_shared_experts", tc.get("num_shared_experts", 1)),
        "router_scaling_factor": tc.get("routed_scaling_factor",
                                        tc.get("router_scaling_factor", 2.0)),
        "route_norm": True,
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "rope_theta": tc.get("rope_theta", rope.get("rope_theta", 5000000.0)),
        "rotary_dim": tc.get("rotary_dim", 64),
        "swiglu_alpha": tc.get("swiglu_alpha", 1.702),
        "swiglu_limit": tc.get("swiglu_limit", 7.0),
        "use_gemma_norm": tc.get("use_gemma_norm", True),
        "eos_token_id": eos,
        "max_position_embeddings": tc.get("max_position_embeddings", 1048576),
        "rope_parameters": {"rope_theta": tc.get("rope_theta", rope.get("rope_theta", 5000000.0))},
    }


def dequant(f, name):
    import torch
    return f.get_tensor(name).to(torch.float32).numpy()


def rename_out(name):
    return SHARED_RE.sub(".mlp.shared_experts.", name)


def should_skip(name, n_layers):
    if SKIP_RE.search(name):
        return True
    li = layer_idx(name)
    if li >= n_layers:
        return True
    if name.startswith("vision_model.") or name.startswith("model.visual."):
        return True
    return False


def classify(name, n_layers):
    if name.endswith("_scale_inv") or name.endswith("_scale"):
        return "consumed"
    if should_skip(name, n_layers):
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
        return "x"
    if name.endswith(".weight"):
        return "q"
    return "f32"


def _split_gate_up(prefix, w):
    two_i, h = w.shape
    i = two_i // 2
    return [
        (prefix + "gate_proj.weight", w[:i]),
        (prefix + "up_proj.weight", w[i:]),
    ]


def iter_tensors(name, w):
    base = name[:-7] if name.endswith(".weight") else name
    if base.endswith(".mlp.experts.gate_up_proj") and w.ndim == 3:
        e, two_i, h = w.shape
        i = two_i // 2
        out = []
        for ex in range(e):
            p = base.replace(".mlp.experts.gate_up_proj", f".mlp.experts.{ex}.")
            gu = w[ex]
            out.append((p + "gate_proj.weight", gu[:i]))
            out.append((p + "up_proj.weight", gu[i:]))
        return out
    if base.endswith(".mlp.experts.down_proj") and w.ndim == 3:
        e = w.shape[0]
        return [
            (base.replace(".mlp.experts.down_proj", f".mlp.experts.{ex}.") + "down_proj.weight", w[ex])
            for ex in range(e)
        ]
    if w.ndim == 2 and base.endswith("gate_up_proj"):
        return _split_gate_up(base.replace("gate_up_proj", ""), w)
    return [(name, w)]


def write_tensor(out_dict, name, w, n_layers, ebits, io_bits, xbits):
    kind = classify(name, n_layers)
    if kind in ("skip", "consumed"):
        return
    if kind == "f32":
        out_dict[name] = w.astype(np.float32)
        return
    bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
    if w.ndim != 2:
        out_dict[name] = w.astype(np.float32)
        return
    q, s = (quant_int2(w, bits) if bits <= 2 else
            quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
    out_dict[name] = q
    out_dict[name + ".qs"] = s


def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            if should_skip(name, n_layers):
                continue
            if name.endswith("_scale_inv") or name.endswith("_scale"):
                continue
            w = dequant(f, name)
            for out_name, arr in iter_tensors(name, w):
                write_tensor(out_dict, rename_out(out_name), arr, n_layers, ebits, io_bits, xbits)


def write_config(src_dir, outdir):
    cfg_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path) as f:
        flat = flatten_config(json.load(f))
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(flat, f, indent=2)


def copy_tokenizer(indir, outdir):
    for fn in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
               "generation_config.json", "chat_template.jinja", "special_tokens_map.json"]:
        src = os.path.join(indir, fn)
        if os.path.exists(src):
            shutil.copy(src, outdir)


def convert_local(indir, outdir, n_layers, ebits, io_bits, xbits):
    from safetensors.numpy import save_file
    os.makedirs(outdir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
    for i, sp in enumerate(shards):
        out = {}
        convert_shard(sp, out, n_layers, ebits, io_bits, xbits)
        save_file(out, os.path.join(outdir, f"out-{i:05d}.safetensors"))
    write_config(indir, outdir)
    copy_tokenizer(indir, outdir)
    print(f"converted {len(shards)} shards -> {outdir}")


def main():
    ap = argparse.ArgumentParser(description="MiniMax-M3 -> Colibri int4 container")
    ap.add_argument("--repo", default="MiniMaxAI/MiniMax-M3")
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=4)
    ap.add_argument("--io-bits", type=int, default=8)
    ap.add_argument("--xbits", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=60)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    a = ap.parse_args()
    if a.xbits is None:
        a.xbits = a.ebits

    if a.indir:
        if not a.outdir:
            sys.exit("--outdir required with --indir")
        convert_local(a.indir, a.outdir, a.n_layers, a.ebits, a.io_bits, a.xbits)
        return

    if not a.outdir:
        sys.exit("--outdir required")

    import convert_fp8_to_int4 as cvt
    cvt.classify = lambda name, n_layers, keep_mtp=False, keep_idx=False: classify(name, n_layers)
    cvt.dequant = dequant
    cvt.convert_shard = convert_shard

    os.makedirs(a.outdir, exist_ok=True)
    from huggingface_hub import hf_hub_download
    try:
        cfg_src = hf_hub_download(a.repo, "config.json", local_dir=a.outdir + "/_meta")
        with open(cfg_src) as f:
            flat = flatten_config(json.load(f))
        with open(os.path.join(a.outdir, "config.json"), "w") as f:
            json.dump(flat, f, indent=2)
    except Exception as ex:
        print(f"warning: config prefetch failed ({ex})")

    argv = [
        "convert_m3.py", "--repo", a.repo, "--outdir", a.outdir,
        "--ebits", str(a.ebits), "--io-bits", str(a.io_bits),
        "--xbits", str(a.xbits), "--n-layers", str(a.n_layers),
        "--min-free-gb", str(a.min_free_gb),
    ]
    sys.argv = argv
    cvt.main()

    meta_cfg = os.path.join(a.outdir, "_meta", "config.json")
    if os.path.exists(meta_cfg):
        write_config(a.outdir + "/_meta", a.outdir)
    tok_dir = a.outdir + "/_meta"
    if os.path.isdir(tok_dir):
        copy_tokenizer(tok_dir, a.outdir)


if __name__ == "__main__":
    main()
