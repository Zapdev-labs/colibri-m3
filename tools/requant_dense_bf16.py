#!/usr/bin/env python3
"""requant_dense_bf16.py — f17 fallback: re-quantize shared experts + dense MLP from BF16.

Downloads shared expert weights (layers 3-59) and dense MLP weights (layers 0-2)
from the HuggingFace BF16 checkpoint using HTTP range requests (~7.7GB total),
quantizes them at per-row int8 from BF16, and overwrites these tensors in the
m3_i4_v3 snapshot.

Shared experts are always-active (1 per MoE layer), so their precision matters
for every token. Dense MLP (first 3 layers) processes every token. Re-quantizing
these from BF16 at int8 (255 levels vs int4's 15) should further improve oracle
match rate beyond the attention int8 fix.

Usage:
  python3 tools/requant_dense_bf16.py --snap /home/ai/models/m3_i4_v3 \
                                      --repo MiniMaxAI/MiniMax-M3

Environment:
  Requires safetensors + numpy + huggingface_hub + torch.
  Use /home/ai/llama-convert-venv/bin/python.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

REPO = "MiniMaxAI/MiniMax-M3"

# Patterns for tensors to re-quantize from BF16.
# Shared experts: language_model.model.layers.N.mlp.shared_mlp.{gate,up,down}_proj.weight
# Dense MLP: language_model.model.layers.N.mlp.{gate,up,down}_proj.weight (only layers 0-2)
SHARED_EXPERT_RE = re.compile(
    r"language_model\.model\.layers\.(\d+)\.(?:block_sparse_moe|mlp)\.shared_experts\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
DENSE_MLP_RE = re.compile(
    r"language_model\.model\.layers\.(\d+)\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def quant_int8(w: np.ndarray):
    """Per-row symmetric int8 quantization (same algorithm as convert.py)."""
    o, i = w.shape
    amax = np.max(np.abs(w), axis=1).astype(np.float64)
    scale64 = np.where(amax > 1e-8, amax / 127.0, 1e-8)
    s = scale64.astype(np.float32)
    scale32 = scale64.astype(np.float32)
    q = np.clip(np.rint(w.astype(np.float32) / scale32[:, None]), -128, 127).astype(np.int8)
    return q, s


def normalize_name(hf_name: str) -> str:
    """Convert HF tensor name to engine name (strip language_model., shared_mlp -> shared_experts)."""
    name = hf_name.removeprefix("language_model.")
    name = name.replace("block_sparse_moe", "mlp")
    name = re.sub(r"\.mlp\.shared_mlp\.", ".mlp.shared_experts.", name)
    return name


def download_tensors_from_shard(repo: str, shard_file: str) -> dict:
    """Download shared expert + dense MLP tensors from one HF shard using range requests."""
    from huggingface_hub import HfFileSystem
    import torch

    fs = HfFileSystem()
    f = fs.open(f"{repo}/{shard_file}", "rb")

    header_len_bytes = f.read(8)
    header_len = int.from_bytes(header_len_bytes, "little")
    header_json = f.read(header_len).decode("utf-8")
    header = json.loads(header_json)
    data_start = 8 + header_len

    results = {}

    for name, info in header.items():
        if name == "__metadata__":
            continue

        is_shared = bool(SHARED_EXPERT_RE.match(name))
        m = DENSE_MLP_RE.match(name)
        is_dense = bool(m) and int(m.group(1)) < 3

        if not is_shared and not is_dense:
            continue

        engine_name = normalize_name(name)

        start, end = info["data_offsets"]
        f.seek(data_start + start)
        raw = f.read(end - start)

        shape = info["shape"]
        dtype_str = info["dtype"]
        if dtype_str == "BF16":
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
            w = tensor.to(torch.float32).numpy()
        elif dtype_str == "F32":
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
            w = tensor.numpy()
        elif dtype_str == "F16":
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape)
            w = tensor.to(torch.float32).numpy()
        else:
            print(f"  WARNING: unexpected dtype {dtype_str} for {name}, skipping")
            continue

        q_i8, scale = quant_int8(w)
        results[engine_name] = (q_i8, scale)

    f.close()
    return results


def process_shard(snap_path: str, tensor_data: dict, dst_path: str) -> dict:
    """Rewrite a snapshot shard with new tensors."""
    from safetensors import safe_open
    from safetensors.numpy import save_file

    out: dict = {}
    n_replaced = 0

    with safe_open(snap_path, framework="numpy") as f:
        keys = list(f.keys())
        for name in keys:
            if name in tensor_data:
                q_i8, scale = tensor_data[name]
                out[name] = q_i8
                out[name + ".qs"] = scale
                n_replaced += 1
            elif name.endswith(".qs") and name[:-3] in tensor_data:
                pass  # already written with the weight tensor
            else:
                arr = f.get_tensor(name)
                out[name] = np.ascontiguousarray(arr)

    save_file(out, dst_path)
    return {
        "shard": os.path.basename(snap_path),
        "replaced": n_replaced,
        "total": len(keys),
        "out_size_gb": os.path.getsize(dst_path) / 1e9,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-quantize shared experts + dense MLP from BF16 (f17 fallback)",
    )
    ap.add_argument("--snap", required=True,
                    help="snapshot dir to update (e.g. /home/ai/models/m3_i4_v3)")
    ap.add_argument("--repo", default=REPO, help="HF repo for BF16 source")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="process only N HF shards (smoke mode)")
    a = ap.parse_args()

    snap = a.snap.rstrip("/")
    if not os.path.isdir(snap):
        sys.exit(f"snapshot dir not found: {snap}")

    from huggingface_hub import list_repo_files
    hf_files = sorted([f for f in list_repo_files(a.repo) if f.endswith(".safetensors")])
    local_shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if a.max_shards is not None:
        hf_files = hf_files[:a.max_shards]
        local_shards = local_shards[:a.max_shards]

    print(f"[bf16-dense] snap={snap}")
    print(f"[bf16-dense] repo={a.repo}")
    print(f"[bf16-dense] HF shards: {len(hf_files)}, local shards: {len(local_shards)}")

    assert len(hf_files) == len(local_shards), \
        f"shard count mismatch: HF={len(hf_files)}, local={len(local_shards)}"

    t0 = time.time()
    total_replaced = 0
    for i, (hf_file, local_path) in enumerate(zip(hf_files, local_shards)):
        ts = time.time()
        tensor_data = download_tensors_from_shard(a.repo, hf_file)
        download_time = time.time() - ts

        if not tensor_data:
            print(f"[{i+1}/{len(hf_files)}] {os.path.basename(local_path)}: "
                  f"no target tensors found, skipping")
            continue

        tmp_path = local_path + ".tmp"
        summary = process_shard(local_path, tensor_data, tmp_path)
        os.replace(tmp_path, local_path)
        total_replaced += summary["replaced"]
        elapsed = time.time() - ts
        print(f"[{i+1}/{len(hf_files)}] {os.path.basename(local_path)}: "
              f"downloaded {len(tensor_data)} tensors ({download_time:.1f}s), "
              f"replaced {summary['replaced']}, "
              f"out={summary['out_size_gb']:.2f} GB, "
              f"elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - t0
    print(f"\n[bf16-dense] done: {total_replaced} tensors replaced")
    print(f"[bf16-dense] total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"[bf16-dense] output: {snap}")

    log_path = os.path.join(snap, "requant_dense_bf16.log")
    with open(log_path, "w") as logf:
        logf.write(f"requant_dense_bf16.py log\n")
        logf.write(f"source: HF {a.repo} (BF16)\n")
        logf.write(f"snap: {snap}\n")
        logf.write(f"tensors replaced: {total_replaced}\n")
        logf.write(f"elapsed: {total_elapsed:.1f}s\n")
        logf.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
