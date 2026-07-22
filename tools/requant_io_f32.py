#!/usr/bin/env python3
"""requant_io_f32.py — f17: upgrade embed_tokens + lm_head from int8 to f32.

Downloads the embedding and lm_head tensors from the HuggingFace BF16
checkpoint (~4.9GB total via HTTP range requests), stores them as f32 in the
snapshot, and removes the per-row int8 .qs scale tensors.

The engine's qt_load auto-detects f32 from the absence of a .qs sidecar
(falls through to the f32 path: fmt=0). So NO engine changes are needed.

lm_head is the FINAL projection producing logits. Upgrading it from int8
(255 levels/row) to f32 (exact) eliminates quantization error in the argmax,
directly reducing greedy-decode token flips at sensitive positions.

Usage:
  python3 tools/requant_io_f32.py --snap /path/to/m3_i4 \
                                  --repo MiniMaxAI/MiniMax-M3

Environment:
  Requires safetensors + numpy + huggingface_hub + torch.
  Use python3.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REPO = "MiniMaxAI/MiniMax-M3"

# Tensors to upgrade from int8 to f32.
IO_TENSORS = {
    "lm_head.weight",
    "model.embed_tokens.weight",
}


def download_io_tensors(repo: str, shard_file: str) -> dict:
    """Download embed_tokens + lm_head from one HF shard as f32 numpy arrays.

    Returns a dict mapping the ENGINE tensor name (language_model. prefix
    stripped) to a float32 numpy array.
    """
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
        engine_name = name.removeprefix("language_model.")
        if engine_name not in IO_TENSORS:
            continue

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

        results[engine_name] = np.ascontiguousarray(w)

    f.close()
    return results


def rewrite_shard(snap_path: str, io_data: dict, dst_path: str) -> dict:
    """Rewrite a snapshot shard, replacing IO tensors with f32 and dropping .qs."""
    from safetensors import safe_open
    from safetensors.numpy import save_file

    out: dict = {}
    n_replaced = 0
    n_dropped_qs = 0

    with safe_open(snap_path, framework="numpy") as f:
        keys = list(f.keys())
        for name in keys:
            if name in io_data:
                # Replace int8 with f32 (no .qs needed — engine auto-detects f32).
                out[name] = io_data[name]
                n_replaced += 1
            elif name.endswith(".qs") and name[:-3] in io_data:
                # Drop the .qs scale — f32 path doesn't use it.
                n_dropped_qs += 1
            else:
                arr = f.get_tensor(name)
                out[name] = np.ascontiguousarray(arr)

    save_file(out, dst_path)
    return {
        "shard": os.path.basename(snap_path),
        "replaced": n_replaced,
        "dropped_qs": n_dropped_qs,
        "total": len(keys),
        "out_size_gb": os.path.getsize(dst_path) / 1e9,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Upgrade embed_tokens + lm_head from int8 to f32 (f17)",
    )
    ap.add_argument("--snap", required=True,
                    help="snapshot dir to update (e.g. /path/to/m3_i4)")
    ap.add_argument("--repo", default=REPO, help="HF repo for BF16 source")
    a = ap.parse_args()

    snap = a.snap.rstrip("/")
    if not os.path.isdir(snap):
        sys.exit(f"snapshot dir not found: {snap}")

    from huggingface_hub import list_repo_files
    hf_files = sorted([f for f in list_repo_files(a.repo) if f.endswith(".safetensors")])
    local_shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))

    print(f"[io-f32] snap={snap}")
    print(f"[io-f32] repo={a.repo}")
    print(f"[io-f32] HF shards: {len(hf_files)}, local shards: {len(local_shards)}")

    assert len(hf_files) == len(local_shards), \
        f"shard count mismatch: HF={len(hf_files)}, local={len(local_shards)}"

    t0 = time.time()
    total_replaced = 0
    for i, (hf_file, local_path) in enumerate(zip(hf_files, local_shards)):
        ts = time.time()
        io_data = download_io_tensors(a.repo, hf_file)
        download_time = time.time() - ts

        if not io_data:
            continue

        tmp_path = local_path + ".tmp"
        summary = rewrite_shard(local_path, io_data, tmp_path)
        os.replace(tmp_path, local_path)
        total_replaced += summary["replaced"]
        elapsed = time.time() - ts
        print(f"[{i+1}/{len(hf_files)}] {os.path.basename(local_path)}: "
              f"downloaded {len(io_data)} IO tensors ({download_time:.1f}s), "
              f"replaced {summary['replaced']}, dropped {summary['dropped_qs']} .qs, "
              f"out={summary['out_size_gb']:.2f} GB, "
              f"elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - t0
    print(f"\n[io-f32] done: {total_replaced} IO tensors upgraded to f32")
    print(f"[io-f32] total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"[io-f32] output: {snap}")

    log_path = os.path.join(snap, "requant_io_f32.log")
    with open(log_path, "w") as logf:
        logf.write(f"requant_io_f32.py log\n")
        logf.write(f"source: HF {a.repo} (BF16)\n")
        logf.write(f"snap: {snap}\n")
        logf.write(f"IO tensors upgraded to f32: {total_replaced}\n")
        logf.write(f"elapsed: {total_elapsed:.1f}s\n")
        logf.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
