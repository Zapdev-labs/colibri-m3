#!/usr/bin/env python3
"""test_numerical_stability.py — VAL-CORR-024: no NaN/Inf in intermediate tensors.

Runs colibri-m3 with M3_CHECK_NAN=1 and DEBUG_TRACE=1 over a 50-token greedy
decode and asserts that no intermediate tensor (hidden states, attention
outputs, MoE outputs, logits) contains NaN or Inf. The engine's nan_check
instrumentation (added in f8-oracle-validation) scans every layer's
rmsnorm_in / attn_out / post_ln / mlp_out / residual and the lm_head logits
at every token position, logging `nan-check: L<N> <kernel> OK` on success or
`nan-check: FAILED ...` on the first NaN/Inf element.

This catches the common MSA-port failure (block scores underflowing softmax
-> NaN) and the common MoE failure (router bias overflow -> Inf).

Usage:
  python3 tools/test_numerical_stability.py [--model /home/ai/models/m3_i4_v2]
                                            [--ngen 50]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGIN = ROOT / "m3"
DEFAULT_MODEL = os.environ.get("COLI_MODEL", "/home/ai/models/m3_i4_v2")


def log(msg: str) -> None:
    print(f"[test_nan] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ngen", type=int, default=50)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "88")))
    ap.add_argument("--debug-trace", action="store_true",
                    help="also enable DEBUG_TRACE=1 to dump per-tensor stats")
    args = ap.parse_args()

    if not ENGIN.is_file():
        log(f"engine binary not found at {ENGIN} — run: make")
        return 1
    env = os.environ.copy()
    env["SNAP"] = args.model
    env["TEMP"] = "0"           # greedy — no sampling nondeterminism
    env["TOPP"] = "0"
    env["TOPK"] = "0"
    env["M3_CHECK_NAN"] = "1"   # instrument every intermediate tensor
    if args.debug_trace:
        env["DEBUG_TRACE"] = "1"
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "close"
    cmd = [str(ENGIN), "128", "4", "8", "4096"]

    try:
        from tokenizers import Tokenizer
    except ImportError:
        log("SKIP: tokenizers lib not installed")
        return 0
    tok_path = Path(args.model) / "tokenizer.json"
    if not tok_path.is_file():
        log(f"SKIP: tokenizer.json not found at {tok_path}")
        return 0
    tok = Tokenizer.from_file(str(tok_path))
    prompt = "Hello, my name is"
    prompt_ids = tok.encode(prompt).ids
    payload = f"{args.ngen}\n" + " ".join(map(str, prompt_ids)) + "\n"
    log(f"launching engine (M3_CHECK_NAN=1, DEBUG_TRACE={1 if args.debug_trace else 0}, "
        f"prompt='{prompt}' ({len(prompt_ids)} toks), ngen={args.ngen})")
    proc = subprocess.run(cmd, input=payload, env=env, capture_output=True,
                          text=True, timeout=7200)
    stderr = proc.stderr or ""
    stdout = proc.stdout or ""
    # Engine returns non-zero if nan failures were detected (the run_gen / run_teacher_force
    # path returns 1 when g_nan_failures > 0 under M3_CHECK_NAN).
    rc = proc.returncode

    # Count nan-check OK / FAILED lines
    ok_lines = [l for l in stderr.splitlines() if l.startswith("nan-check:") and " OK" in l]
    fail_lines = [l for l in stderr.splitlines() if l.startswith("nan-check:") and "FAILED" in l]
    trace_lines = [l for l in stderr.splitlines() if l.startswith("trace:")]
    # also surface the engine's summary
    summary_lines = [l for l in stderr.splitlines() if "[nan-check]" in l]

    # count emitted tokens
    emitted = 0
    for line in stdout.splitlines():
        line = line.strip()
        if line.isdigit() or (line.startswith("-") and line[1:].isdigit()):
            emitted += 1

    log(f"engine rc={rc} | tokens emitted={emitted} | nan-check OK={len(ok_lines)} "
        f"FAILED={len(fail_lines)} | trace lines={len(trace_lines)}")
    for s in summary_lines:
        log(f"  {s}")
    if fail_lines:
        log("first 10 failures:")
        for l in fail_lines[:10]:
            log(f"  {l}")

    passed = (len(fail_lines) == 0 and rc == 0)
    if passed:
        log(f"PASS: no NaN/Inf detected across {len(ok_lines)} tensor checks over "
            f"{emitted} tokens")
    else:
        log(f"FAIL: {len(fail_lines)} NaN/Inf failures detected (rc={rc})")

    # also verify a debug-trace dump was produced if requested
    trace_ok = True
    if args.debug_trace:
        if len(trace_lines) == 0:
            log("WARN: --debug-trace requested but no trace: lines in output")
            trace_ok = False
        else:
            log(f"  debug-trace: {len(trace_lines)} tensor stats lines dumped")
            # show a sample
            for l in trace_lines[:3]:
                log(f"    {l}")

    result_path = ROOT / "tests" / "oracle" / "nan_test_result.json"
    result_path.write_text(json.dumps({
        "passed": passed,
        "tokens_emitted": emitted,
        "nan_check_ok": len(ok_lines),
        "nan_check_failed": len(fail_lines),
        "trace_lines": len(trace_lines),
        "debug_trace": args.debug_trace,
        "engine_rc": rc,
        "first_failures": fail_lines[:10],
    }, indent=2))
    log(f"wrote {result_path}")
    return 0 if (passed and trace_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
