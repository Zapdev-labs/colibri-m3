#!/usr/bin/env python3
"""requant_attn_bf16.py — f17 fallback: re-quantize attention from original BF16.

Downloads ONLY the attention projection tensors from the HuggingFace BF16
checkpoint (MiniMaxAI/MiniMax-M3) using HTTP range requests (~12.6GB total
instead of 852GB), quantizes them at per-row int8 directly from BF16, and
overwrites the attention tensors in the m3_i4_v3 snapshot.

This is the fallback approach when int4→int8 re-quantization (requant_attn.py)
doesn't improve the oracle match rate. The root cause is that int4→int8 only
preserves the 15 int4 levels; true int8 from BF16 gives 255 levels per row,
a 17× improvement in quantization granularity.

The engine's qt_load auto-detects int8 from nbytes == O*I, so no engine
changes are needed.

Usage:
  python3 tools/requant_attn_bf16.py --snap /home/ai/models/m3_i4_v3 \
                                     --repo MiniMaxAI/MiniMax-M3

Environment:
  Requires safetensors + numpy + huggingface_hub + torch (for BF16 loading).
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

# Attention projection tensor name pattern (in the HF BF16 checkpoint).
# HF stores these as language_model.model.layers.N.self_attn.{q,k,v,o}_proj.weight
# and language_model.model.layers.N.self_attn.index_{q,k}_proj.weight
ATTN_RE = re.compile(
    r"language_model\.model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj|index_q_proj|index_k_proj)\.weight$"
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


def download_attn_tensors_from_shard(repo: str, shard_file: str) -> dict:
    """Download attention tensors from one HF shard using range requests.

    Returns a dict mapping the ENGINE tensor name to (int8_q, float32_scale).
    """
    from huggingface_hub import HfFileSystem
    import torch

    fs = HfFileSystem()
    f = fs.open(f"{repo}/{shard_file}", "rb")

    # Read safetensors header.
    header_len_bytes = f.read(8)
    header_len = int.from_bytes(header_len_bytes, "little")
    header_json = f.read(header_len).decode("utf-8")
    header = json.loads(header_json)
    data_start = 8 + header_len

    results = {}

    for name, info in header.items():
        if name == "__metadata__":
            continue
        m = ATTN_RE.match(name)
        if not m:
            continue

        layer_idx = int(m.group(1))
        proj_type = m.group(2)

        # Convert HF name to engine name (strip language_model. prefix).
        engine_name = name.removeprefix("language_model.")

        # Skip index_q_proj/index_k_proj for dense layers (0-2).
        if proj_type in ("index_q_proj", "index_k_proj") and layer_idx < 3:
            continue

        # Read the tensor bytes.
        start, end = info["data_offsets"]
        f.seek(data_start + start)
        raw = f.read(end - start)

        # Convert BF16 bytes to float32 via torch.
        shape = info["shape"]
        dtype_str = info["dtype"]
        if dtype_str == "BF16":
            tensor = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape)
            w = tensor.to(torch.float32).numpy()
        elif dtype_str == "F32":
            tensor = torch.frombuffer(raw, dtype=torch.float32).reshape(shape)
            w = tensor.numpy()
        elif dtype_str == "F16":
            tensor = torch.frombuffer(raw, dtype=torch.float16).reshape(shape)
            w = tensor.to(torch.float32).numpy()
        else:
            print(f"  WARNING: unexpected dtype {dtype_str} for {name}, skipping")
            continue

        # Quantize at int8.
        q_i8, scale = quant_int8(w)
        results[engine_name] = (q_i8, scale)

    f.close()
    return results


def process_shard(snap_path: str, attn_data: dict, dst_path: str) -> dict:
    """Rewrite a snapshot shard with new attention tensors.

    Loads all tensors from snap_path, replaces attention weights + .qs scales
    with the BF16-derived int8 versions, and saves to dst_path.
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file

    out: dict = {}
    n_replaced = 0

    with safe_open(snap_path, framework="numpy") as f:
        keys = list(f.keys())
        for name in keys:
            if name in attn_data:
                # Replace with BF16-derived int8.
                q_i8, scale = attn_data[name]
                out[name] = q_i8
                out[name + ".qs"] = scale
                n_replaced += 1
            elif name.endswith(".qs") and name[:-3] in attn_data:
                # This .qs was already written with the weight tensor. Skip.
                pass
            else:
                # Copy as-is.
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
        description="Re-quantize attention from BF16 source (f17 fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    # Get list of HF shards.
    from huggingface_hub import list_repo_files
    hf_files = sorted([f for f in list_repo_files(a.repo) if f.endswith(".safetensors")])
    if a.max_shards is not None:
        hf_files = hf_files[:a.max_shards]
    total_hf = len(hf_files)

    # Get list of local snapshot shards.
    local_shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if a.max_shards is not None:
        local_shards = local_shards[:a.max_shards]
    print(f"[bf16-requant] snap={snap}")
    print(f"[bf16-requant] repo={a.repo}")
    print(f"[bf16-requant] HF shards: {total_hf}, local shards: {len(local_shards)}")

    # Map HF shard index to local shard. Both should be 59 shards.
    # HF: model-00001-of-00059.safetensors
    # Local: out-00000-of-00059.safetensors
    assert len(hf_files) == len(local_shards), \
        f"shard count mismatch: HF={len(hf_files)}, local={len(local_shards)}"

    t0 = time.time()
    total_replaced = 0
    for i, (hf_file, local_path) in enumerate(zip(hf_files, local_shards)):
        ts = time.time()
        # Download attention tensors from HF.
        attn_data = download_attn_tensors_from_shard(a.repo, hf_file)
        download_time = time.time() - ts

        # Rewrite the local shard with new attention tensors.
        # Write to a temp file, then rename.
        tmp_path = local_path + ".tmp"
        summary = process_shard(local_path, attn_data, tmp_path)
        os.replace(tmp_path, local_path)
        total_replaced += summary["replaced"]
        elapsed = time.time() - ts
        print(f"[{i+1}/{total_hf}] {os.path.basename(local_path)}: "
              f"downloaded {len(attn_data)} attn tensors ({download_time:.1f}s), "
              f"replaced {summary['replaced']}, "
              f"out={summary['out_size_gb']:.2f} GB, "
              f"elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - t0
    print(f"\n[bf16-requant] done: {total_hf} shards, {total_replaced} attn tensors replaced")
    print(f"[bf16-requant] total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"[bf16-requant] output: {snap}")

    # Update the requant log.
    log_path = os.path.join(snap, "requant_attn.log")
    with open(log_path, "w") as logf:
        logf.write(f"requant_attn_bf16.py log\n")
        logf.write(f"source: HF {a.repo} (BF16)\n")
        logf.write(f"snap: {snap}\n")
        logf.write(f"shards: {total_hf}\n")
        logf.write(f"attn tensors replaced: {total_replaced}\n")
        logf.write(f"elapsed: {total_elapsed:.1f}s\n")
        logf.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
