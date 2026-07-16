#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${COLI_MODEL:-/home/ai/models/m3_i4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-96}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export COLI_MODEL="$MODEL"
export PLANAR_KV="${PLANAR_KV:-1}"
export PIPE="${PIPE:-1}"
export IDOT="${IDOT:-1}"
export TEMP="${TEMP:-1.0}"
export NUCLEUS="${NUCLEUS:-0.95}"
cd "$ROOT/c"
exec ./coli "$@"
