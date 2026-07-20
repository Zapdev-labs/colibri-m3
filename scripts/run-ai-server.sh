#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COLI_MODEL="${COLI_MODEL:-/home/ai/models/m3_i4_v3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-96}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
cd "$ROOT"
exec ./coli --planar --cap 256 --ctx 8192 "$@"
