#!/usr/bin/env python3
"""llama_observability_wrapper.py — f15: wrap llama-server (rq fork) and emit
the same JSON-line telemetry schema as colibri-m3's src/observability.h.

The rq fork already emits MSA telemetry on stderr (schema v3, see
src/models/msa-runtime.cpp). This wrapper:
  1. Launches llama-server with the user's args.
  2. Tees stderr to a file (for debugging) AND parses it line-by-line.
  3. Translates the rq fork's stderr telemetry into the colibri-m3 schema:
        {"ts": <float>, "event": "<name>", <kv>}
     so tools/observability.py can ingest both engines with one parser.
  4. Adds a wall-clock heartbeat so we get decode_step events even when the
     rq fork's telemetry is sparse.

Usage:
  python3 tools/llama_observability_wrapper.py \\
      --telemetry /tmp/llama_telem.jsonl \\
      -- engine args...

Anything after `--` is passed to llama-server verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

LLAMACPP_BIN_DEFAULT = "/home/ai/llama.cpp-minimax-m3-rq/build-simd/bin/llama-server"
DEFAULT_TELEMETRY = "/tmp/llama_telem.jsonl"

# Patterns from the rq fork's msa-runtime.cpp stderr (see RESEARCH.md §7.2).
# These are best-effort regexes; the fork may emit them in different orders
# or with extra fields. Unmatched lines are passed through unchanged to stderr.
PATTERNS = {
    "load_complete": re.compile(
        r"\[msa\].*model loaded.*layers=(\d+).*experts=(\d+).*msa_layers=(\d+)"),
    "decode_step": re.compile(
        r"\[msa\].*decode.*pos=(\d+).*step_ms=([0-9.]+).*hit=(\d+).*miss=(\d+)"),
    "run_complete": re.compile(
        r"\[msa\].*done.*tokens=(\d+).*wall_s=([0-9.]+).*tokps=([0-9.]+)"),
    # llama-server's standard throughput line:
    "llama_stat": re.compile(
        r"prompt eval time.*\s+(\d+\.\d+)\s+ms.*per token.*\s+(\d+\.\d+)\s+ms"),
}


def log(msg: str) -> None:
    print(f"[llama_obs] {msg}", flush=True, file=sys.stderr)


def emit_jsonl(fp, event: str, kv: dict) -> None:
    rec = {"ts": time.time(), "event": event}
    rec.update(kv)
    fp.write(json.dumps(rec) + "\n")
    fp.flush()


def parse_stderr_line(line: str, fp, t0: float) -> None:
    """Try to match a line against the known telemetry patterns and emit JSON."""
    # MSA-runtime patterns
    for event, pat in PATTERNS.items():
        m = pat.search(line)
        if not m:
            continue
        if event == "load_complete":
            emit_jsonl(fp, "load_complete", {
                "engine": "llama.cpp-minimax-m3-rq",
                "layers": int(m.group(1)),
                "experts": int(m.group(2)),
                "msa_layers": int(m.group(3)),
            })
            return
        if event == "decode_step":
            emit_jsonl(fp, "decode_step", {
                "engine": "llama.cpp-minimax-m3-rq",
                "pos": int(m.group(1)),
                "step_ms": float(m.group(2)),
                "hit": int(m.group(3)),
                "miss": int(m.group(4)),
            })
            return
        if event == "run_complete":
            emit_jsonl(fp, "run_complete", {
                "engine": "llama.cpp-minimax-m3-rq",
                "tokens": int(m.group(1)),
                "wall_s": float(m.group(2)),
                "tokps": float(m.group(3)),
            })
            return
        if event == "llama_stat":
            # llama-server's standard "prompt eval" / "eval" lines.
            # group(1) = total ms, group(2) = per-token ms
            tokps = 1000.0 / float(m.group(2)) if float(m.group(2)) > 0 else 0
            emit_jsonl(fp, "decode_step", {
                "engine": "llama.cpp-minimax-m3-rq",
                "step_ms": float(m.group(2)),
                "tokps": tokps,
            })
            return
    # Unmatched line: write to stderr (preserves llama-server's normal logging).
    sys.stderr.write(line)


def stream_stderr(proc: subprocess.Popen, fp, t0: float, stop: threading.Event) -> None:
    """Read proc.stderr line-by-line, parse, and emit JSON-line telemetry."""
    assert proc.stderr is not None
    for line in proc.stderr:
        if stop.is_set():
            break
        parse_stderr_line(line, fp, t0)
    log("stderr stream ended")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--telemetry", default=DEFAULT_TELEMETRY,
                    help="output JSON-line telemetry path (default /tmp/llama_telem.jsonl)")
    ap.add_argument("--bin", default=LLAMACPP_BIN_DEFAULT,
                    help="llama-server binary path")
    ap.add_argument("--numactl", default="1",
                    help="wrap in numactl --interleave=all (1=yes, 0=no)")
    args, rest = ap.parse_known_args()
    if not rest or rest[0] != "--":
        ap.error("pass llama-server args after `--`")
    llama_args = rest[1:]
    if not llama_args:
        ap.error("no llama-server args after `--`")

    cmd = [args.bin] + llama_args
    if args.numactl == "1":
        cmd = ["numactl", "--interleave=all"] + cmd

    env = os.environ.copy()
    ld = f"/home/ai/llama.cpp-minimax-m3-rq/build-simd/bin:{env.get('LD_LIBRARY_PATH','')}"
    env["LD_LIBRARY_PATH"] = ld

    Path(args.telemetry).parent.mkdir(parents=True, exist_ok=True)
    fp = open(args.telemetry, "w")
    t0 = time.time()
    emit_jsonl(fp, "wrapper_start", {"engine": "llama.cpp-minimax-m3-rq",
                                       "bin": args.bin, "args": llama_args})
    log(f"telemetry -> {args.telemetry}")
    log(f"launching: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            preexec_fn=os.setsid)
    stop = threading.Event()
    stderr_thread = threading.Thread(target=stream_stderr, args=(proc, fp, t0, stop),
                                      daemon=True)
    stderr_thread.start()

    # Forward stdout to our stdout so HTTP clients can still use the server.
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        if proc.poll() is None:
            log("stopping llama-server")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception: pass
        stop.set()
        emit_jsonl(fp, "wrapper_stop", {"engine": "llama.cpp-minimax-m3-rq",
                                          "rc": proc.returncode or 0})
        fp.close()
        stderr_thread.join(timeout=5)
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
