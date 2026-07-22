#!/usr/bin/env bash
# run-bench.sh — f13: throughput benchmark driver.
#
# Runs colibri-m3 with the f13 throughput benchmark, optionally with VNNI
# kernels (f11), NUMA interleave (f10), parallel MoE dispatch (f12), and
# telemetry (f15) enabled. Captures results to tests/oracle/.
#
# Usage:
#   scripts/run-bench.sh                           # default: VNNI+NUMA, 200 tok
#   NGEN=500 scripts/run-bench.sh                  # longer run
#   USE_VNNI=0 scripts/run-bench.sh                # disable VNNI
#   TELEMETRY=/tmp/colibri_telem.jsonl scripts/run-bench.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NGEN="${NGEN:-200}"
USE_VNNI="${USE_VNNI:-1}"
USE_MOE_PARALLEL="${USE_MOE_PARALLEL:-1}"
TELEMETRY="${TELEMETRY:-}"
PROMPT="${PROMPT:-Once upon a time there lived a humble programmer named Alice who}"

ARGS=(--ngen "$NGEN" --prompt "$PROMPT")
[ "$USE_VNNI" = "1" ] && ARGS+=(--use-vnni)
[ -n "$TELEMETRY" ] && ARGS+=(--telemetry "$TELEMETRY")

export USE_VNNI="$USE_VNNI"
export MOE_PARALLEL="$USE_MOE_PARALLEL"
export NUMA_INTERLEAVE=1
export NUMA_PIN_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-88}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "[run-bench] launching bench_throughput.py with: ${ARGS[*]}"
python3 tools/bench_throughput.py "${ARGS[@]}"
