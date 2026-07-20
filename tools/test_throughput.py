#!/usr/bin/env python3
"""test_throughput.py — f13: lightweight smoke test for bench_throughput.py.

Validates the harness structure without running the full 199GB model.
Checks:
  - bench_throughput.py imports cleanly.
  - The parse_engine_stderr() function extracts the right fields from a
    sample stderr.
  - The numastat_delta() function returns a dict.
  - The constants match the f13 contract (TARGET_TOKPS=5.0, etc.).

Usage:
  python3 tools/test_throughput.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[test_throughput] {msg}", flush=True)


SAMPLE_STDERR = """[prefill] done in 12.50s
[stat] 200 tok in 38.50s (5.19 tok/s) | expert hit 96.2%
"""


def main() -> int:
    rc = 0
    spec = importlib.util.spec_from_file_location(
        "bench_throughput", str(ROOT / "tools" / "bench_throughput.py"))
    if not spec or not spec.loader:
        log("FAIL: could not load bench_throughput.py spec")
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log("OK: bench_throughput.py imports cleanly")

    # Check constants.
    if mod.TARGET_TOKPS != 5.0:
        log(f"FAIL: TARGET_TOKPS should be 5.0, got {mod.TARGET_TOKPS}")
        rc = 1
    else:
        log(f"OK: TARGET_TOKPS = {mod.TARGET_TOKPS}")
    if mod.TARGET_HIT_RATE != 0.95:
        log(f"FAIL: TARGET_HIT_RATE should be 0.95, got {mod.TARGET_HIT_RATE}")
        rc = 1
    else:
        log(f"OK: TARGET_HIT_RATE = {mod.TARGET_HIT_RATE}")

    # Check parse_engine_stderr.
    stats = mod.parse_engine_stderr(SAMPLE_STDERR)
    if stats.get("prefill_s") != 12.5:
        log(f"FAIL: prefill_s should be 12.5, got {stats.get('prefill_s')}")
        rc = 1
    else:
        log(f"OK: prefill_s parsed = {stats['prefill_s']}")
    if stats.get("tokens") != 200:
        log(f"FAIL: tokens should be 200, got {stats.get('tokens')}")
        rc = 1
    elif stats.get("tokps_overall") != 5.19:
        log(f"FAIL: tokps_overall should be 5.19, got {stats.get('tokps_overall')}")
        rc = 1
    else:
        log(f"OK: tokps_overall parsed = {stats['tokps_overall']}, tokens = {stats['tokens']}")
    if stats.get("hit_rate_pct") != 96.2:
        log(f"FAIL: hit_rate_pct should be 96.2, got {stats.get('hit_rate_pct')}")
        rc = 1
    else:
        log(f"OK: hit_rate_pct parsed = {stats['hit_rate_pct']}")

    # Check numastat_delta (may fail if numastat not installed; that's OK).
    delta = mod.numastat_delta("colibri")
    if "error" in delta:
        log(f"OK: numastat_delta returned error (numastat may be unavailable): {delta['error']}")
    else:
        log(f"OK: numastat_delta returned {list(delta.keys())}")

    print(f"[test_throughput] overall {'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
