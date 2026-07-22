#!/usr/bin/env python3
"""requant_attn.py — f17: re-quantize attention projections from int4 to int8.

Reads the m3_i4_v2 snapshot (per-row int4 attention tensors), dequantizes them
to f32, re-quantizes at per-row int8, and writes the result to m3_i4_v3.
Only the attention projection tensors are changed:

    *.self_attn.{q,k,v,o}_proj.weight
    *.self_attn.index_{q,k}_proj.weight

All other tensors (experts, norms, routers, embeddings, lm_head, dense MLP)
are copied byte-for-byte from v2. This adds ~3GB to the snapshot (~55MB/layer
at int4 -> ~110MB/layer at int8, 60 layers) but directly improves attention
score accuracy — the most sensitive part of the model.

The engine's ``qt_load`` auto-detects int8 vs int4 from the on-disk byte count
(``nb == O * I`` => int8, else int4), so NO engine code changes are needed:
the int8 attention tensors are loaded via the existing ``matmul_i8`` path
(the same path used for embed_tokens/lm_head).

Usage:
  python3 tools/requant_attn.py --src /path/to/m3_i4 \
                               --dst /path/to/m3_i4

Environment:
  Requires safetensors + numpy (e.g. python3).
  Does NOT require torch (uses safetensors numpy framework).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np

# Attention projection tensor name pattern. Matches the weight tensor (not .qs).
# q_proj, k_proj, v_proj, o_proj, index_q_proj, index_k_proj.
ATTN_WEIGHT_RE = re.compile(
    r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|index_q_proj|index_k_proj)\.weight$"
)

# Sidecar files to copy verbatim from src to dst.
SIDECAR_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "convert.log",
)


def dequant_int4(packed_u8: np.ndarray, scales: np.ndarray, O: int, I: int) -> np.ndarray:
    """Unpack per-row int4 nibbles and dequantize to f32.

    Mirrors the packing in convert.py's quant_int4():
      - Each row is packed two-nibbles-per-byte (low nibble first).
      - Values are stored as unsigned (q + 8), so subtract 8 to get [-8, 7].
      - If I is odd, the trailing high nibble is 8 (neutral / 0 signed).

    Args:
        packed_u8: [O, (I+1)//2] uint8 array of packed nibbles.
        scales: [O] float32 per-row scales.
        O, I: output dimensions (O rows, I columns).

    Returns:
        [O, I] float32 dequantized weights.
    """
    rb = (I + 1) // 2
    assert packed_u8.shape == (O, rb), f"expected ({O}, {rb}), got {packed_u8.shape}"
    assert scales.shape == (O,), f"expected ({O},), got {scales.shape}"

    lo = (packed_u8 & 0x0F).astype(np.int32)  # [O, rb]
    hi = (packed_u8 >> 4).astype(np.int32)     # [O, rb]

    # Interleave: q[0]=lo[0], q[1]=hi[0], q[2]=lo[1], q[3]=hi[1], ...
    q = np.empty((O, rb * 2), dtype=np.int32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    q = q[:, :I]  # trim padding nibble if I is odd

    # Subtract 8 to get signed values in [-8, 7].
    q -= 8

    # Dequantize: w = q * scale_per_row.
    w = q.astype(np.float32) * scales.astype(np.float32)[:, None]
    return w


def quant_int8(w: np.ndarray):
    """Per-row symmetric int8 quantization (same algorithm as convert.py).

    Args:
        w: [O, I] float32 weights.

    Returns:
        (q_int8 [O, I] int8, scales [O] float32).
    """
    o, i = w.shape
    # Match convert.py: amax computed in float32, upcast to float64 for scale.
    amax = np.max(np.abs(w), axis=1).astype(np.float64)
    scale64 = np.where(amax > 1e-8, amax / 127.0, 1e-8)
    s = scale64.astype(np.float32)
    # Divide in float32 to match convert.py's promotion behavior exactly.
    scale32 = scale64.astype(np.float32)
    q = np.clip(np.rint(w.astype(np.float32) / scale32[:, None]), -128, 127).astype(np.int8)
    return q, s


def is_attn_weight(name: str) -> bool:
    """True if this tensor is an attention projection weight (not the .qs scale)."""
    return bool(ATTN_WEIGHT_RE.search(name)) and not name.endswith(".qs")


def is_attn_scale(name: str) -> bool:
    """True if this tensor is the .qs scale for an attention projection weight."""
    if not name.endswith(".qs"):
        return False
    base = name[:-3]  # strip ".qs"
    return bool(ATTN_WEIGHT_RE.search(base))


def process_shard(src_path: str, dst_path: str) -> dict:
    """Process one shard: copy all tensors, re-quantizing attention weights.

    Returns a summary dict with counts of tensors processed/requantized.
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file

    out: dict = {}
    n_total = 0
    n_requant = 0
    n_copied = 0
    requant_names = []

    with safe_open(src_path, framework="numpy") as f:
        keys = list(f.keys())
        for name in keys:
            n_total += 1
            arr = f.get_tensor(name)

            if is_attn_weight(name):
                # This is an int4-packed attention weight.
                # Shape is [O, (I+1)//2] uint8. Recover I from the .qs tensor
                # and the known attention dimensions.
                # The stored shape is [O, I_packed]. I = I_packed * 2 (or 2*rb-1
                # if original I was odd, but attention dims are always even).
                O, I_packed = arr.shape
                I = I_packed * 2

                # Read the corresponding scale tensor.
                qs_name = name + ".qs"
                with safe_open(src_path, framework="numpy") as f2:
                    scales = f2.get_tensor(qs_name)

                # Dequantize int4 -> f32.
                w_f32 = dequant_int4(arr, scales, O, I)

                # Re-quantize at int8.
                q_i8, s_new = quant_int8(w_f32)

                out[name] = q_i8
                out[qs_name] = s_new
                n_requant += 1
                requant_names.append(name)

            elif is_attn_scale(name):
                # This is the .qs scale for an attention weight.
                # It was already written when we processed the corresponding
                # weight tensor above. Skip it here to avoid overwriting.
                n_copied += 1
                # Don't add to out — already handled by the weight branch.

            else:
                # Pass-through: copy as-is.
                # Ensure contiguous array for safetensors save.
                out[name] = np.ascontiguousarray(arr)
                n_copied += 1

    save_file(out, dst_path)
    return {
        "shard": os.path.basename(src_path),
        "total": n_total,
        "requantized": n_requant,
        "copied": n_copied,
        "requant_names": requant_names,
        "out_size_gb": os.path.getsize(dst_path) / 1e9,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-quantize attention projections from int4 to int8 (f17)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", required=True, help="source snapshot dir (e.g. /path/to/m3_i4)")
    ap.add_argument("--dst", required=True, help="destination snapshot dir (e.g. /path/to/m3_i4)")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="process only N shards (smoke mode for testing)")
    a = ap.parse_args()

    src = a.src.rstrip("/")
    dst = a.dst.rstrip("/")

    if not os.path.isdir(src):
        sys.exit(f"source dir not found: {src}")

    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not shards:
        sys.exit(f"no .safetensors shards in {src}")

    if a.max_shards is not None:
        shards = shards[:a.max_shards]

    total_shards = len(shards)
    print(f"[requant] src={src}")
    print(f"[requant] dst={dst}")
    print(f"[requant] shards to process: {total_shards}")

    os.makedirs(dst, exist_ok=True)

    # Copy sidecar files (config, tokenizer, etc.)
    for fn in SIDECAR_FILES:
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, fn))
            print(f"[requant] copied sidecar: {fn}")

    # Process each shard.
    t0 = time.time()
    total_requant = 0
    for i, sp in enumerate(shards):
        shard_name = os.path.basename(sp)
        dst_path = os.path.join(dst, shard_name)
        ts = time.time()
        summary = process_shard(sp, dst_path)
        total_requant += summary["requantized"]
        elapsed = time.time() - ts
        print(f"[{i+1}/{total_shards}] {shard_name}: "
              f"requantized {summary['requantized']} attn tensors, "
              f"copied {summary['copied']}, "
              f"out={summary['out_size_gb']:.2f} GB, "
              f"elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - t0
    print(f"\n[requant] done: {total_shards} shards, {total_requant} attn tensors re-quantized")
    print(f"[requant] total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"[requant] output: {dst}")

    # Write a requant log.
    log_path = os.path.join(dst, "requant_attn.log")
    with open(log_path, "w") as logf:
        logf.write(f"requant_attn.py log\n")
        logf.write(f"src: {src}\n")
        logf.write(f"dst: {dst}\n")
        logf.write(f"shards: {total_shards}\n")
        logf.write(f"attn tensors re-quantized: {total_requant}\n")
        logf.write(f"elapsed: {total_elapsed:.1f}s\n")
        logf.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
