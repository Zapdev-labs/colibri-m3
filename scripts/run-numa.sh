#!/usr/bin/env bash
# run-numa.sh — f10: NUMA-aware deployment wrapper for colibri-m3.
#
# Implements the documented Cascade Lake NUMA recipe (RESEARCH.md §5.7):
#   1. Disable kernel NUMA balancing (best-effort, needs root).
#   2. Wrap engine in numactl --interleave=all so model pages spread evenly.
#   3. Pin OMP threads to physical cores per NUMA node.
#   4. Leave 8 cores for OS/SSH (96 - 8 = 88 worker threads).
#
# Usage:
#   scripts/run-numa.sh chat --planar --cap 256 --ctx 8192
#   scripts/run-numa.sh serve --host 0.0.0.0 --port 8080
#   NUMA_PIN_THREADS=0 scripts/run-numa.sh bench    # disable pinning
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Best-effort: disable NUMA balancing (root only).
if [ "$(id -u)" = "0" ]; then
    echo 0 > /proc/sys/kernel/numa_balancing 2>/dev/null || true
fi

# Default to 88 worker threads on the dual-Xeon-5220R host (96 - 8 OS cores).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-88}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

# Engine reads these env vars (added in f10):
export NUMA_INTERLEAVE="${NUMA_INTERLEAVE:-1}"
export NUMA_PIN_THREADS="${NUMA_PIN_THREADS:-1}"

# Use numactl --interleave=all for the outer process so the engine's mallocs
# (model struct, KV cache, expert slab) spread across both NUMA nodes.
NUMACTL_BIN="${NUMACTL_BIN:-$(command -v numactl || true)}"
if [ -n "$NUMACTL_BIN" ]; then
    exec "$NUMACTL_BIN" --interleave=all "$ROOT/coli" "$@"
else
    # No numactl: fall back to the engine's internal mbind() path (still
    # interleaves via NUMA_INTERLEAVE=1, just not the outer process).
    exec "$ROOT/coli" "$@"
fi
