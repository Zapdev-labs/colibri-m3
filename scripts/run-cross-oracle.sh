#!/usr/bin/env bash
# run-cross-oracle.sh — f9: cross-oracle comparison driver.
#
# Runs colibri-m3 and the llama.cpp-minimax-m3-rq fork on identical prompts
# and reports greedy + teacher-forcing agreement. Contract (VAL-CORR-021):
# >=19/20 greedy token IDs match across the two independent int4 quantization
# schemes.
#
# Usage:
#   scripts/run-cross-oracle.sh                       # default prompt
#   PROMPT="The capital of France is" scripts/run-cross-oracle.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog}"
NGEN="${NGEN:-20}"
TF_TOKENS="${TF_TOKENS:-32}"

export COLI_MODEL="${COLI_MODEL:-/path/to/m3_i4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-88}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export USE_VNNI="${USE_VNNI:-1}"
export NUMA_INTERLEAVE=1
export NUMA_PIN_THREADS=1

python3 tools/cross_oracle_compare.py \
    --mode both \
    --prompt "$PROMPT" \
    --ngen "$NGEN" \
    --tf-tokens "$TF_TOKENS" \
    --keep-server
