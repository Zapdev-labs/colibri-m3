#!/usr/bin/env python3
"""test_determinism.py — VAL-CORR-018 & VAL-CROSS-006: seeded determinism.

Verifies that two consecutive `coli run` invocations with the same seed and
prompt produce byte-identical token output. This validates the full
determinism chain: seeded RNG (srand with a fixed seed), no hidden
nondeterminism in thread scheduling or floating-point reduction order, no
time-based seeds in sampling.

Two modes are tested:

  (A) VAL-CORR-018: greedy determinism (TEMP=0 TOPP=0 TOPK=0). No RNG at all
      — the engine must produce identical output because the argmax path is
      fully deterministic. This is the strictest check.

  (B) VAL-CROSS-006: seeded sampling determinism (--seed 424242, TEMP=1.0).
      The engine uses rand() for sampling; with a fixed seed, two runs must
      produce the same token sequence. This catches time-seeding bugs.

Usage:
  python3 tools/test_determinism.py [--model /home/ai/models/m3_i4_v3]
                                    [--seed 424242] [--prompt "The capital of France is"]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGIN = ROOT / "m3"
DEFAULT_MODEL = os.environ.get("COLI_MODEL", "/home/ai/models/m3_i4_v3")
DEFAULT_SEED = 424242
DEFAULT_PROMPT = "The capital of France is"


def log(msg: str) -> None:
    print(f"[test_determinism] {msg}", file=sys.stderr, flush=True)


def run_once(model: str, prompt_ids: list[int], ngen: int, seed: int | None,
             greedy: bool, threads: int) -> tuple[str, str, int]:
    """Run the engine once; return (stdout_token_text, stderr_tail, returncode)."""
    if not ENGIN.is_file():
        log(f"engine binary not found at {ENGIN} — run: make")
        sys.exit(1)
    env = os.environ.copy()
    env["SNAP"] = model
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "close"
    if greedy:
        env["TEMP"] = "0"
        env["TOPP"] = "0"
        env["TOPK"] = "0"
    else:
        env["TEMP"] = "1.0"
        env["TOPP"] = "0.95"
        env["TOPK"] = "40"
        if seed is not None:
            env["SEED"] = str(seed)
    cmd = [str(ENGIN), "128", "4", "8", "4096"]
    payload = f"{ngen}\n" + " ".join(map(str, prompt_ids)) + "\n"
    proc = subprocess.run(cmd, input=payload, env=env, capture_output=True,
                          text=True, timeout=7200)
    return proc.stdout, proc.stderr or "", proc.returncode


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--ngen", type=int, default=50)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "88")))
    args = ap.parse_args()

    # Tokenize the prompt
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
    prompt_ids = tok.encode(args.prompt).ids
    log(f"prompt='{args.prompt}' -> {len(prompt_ids)} tokens")

    all_passed = True

    # --- (A) VAL-CORR-018: greedy determinism ---
    log("--- (A) greedy determinism (VAL-CORR-018) ---")
    out1, err1, rc1 = run_once(args.model, prompt_ids, args.ngen, None, True, args.threads)
    out2, err2, rc2 = run_once(args.model, prompt_ids, args.ngen, None, True, args.threads)
    if rc1 != 0 or rc2 != 0:
        log(f"FAIL: engine exited {rc1}/{rc2}")
        sys.stderr.write(err1[-2000:] + "\n---\n" + err2[-2000:])
        all_passed = False
    elif out1 == out2:
        log(f"PASS (A): greedy runs byte-identical (sha {sha(out1)})")
    else:
        log(f"FAIL (A): greedy runs differ (sha1={sha(out1)} sha2={sha(out2)})")
        # show first divergence
        l1, l2 = out1.splitlines(), out2.splitlines()
        for i in range(min(len(l1), len(l2))):
            if l1[i] != l2[i]:
                log(f"  first divergence at line {i}: run1={l1[i]!r} run2={l2[i]!r}")
                break
        all_passed = False

    # --- (B) VAL-CROSS-006: seeded sampling determinism ---
    log(f"--- (B) seeded sampling determinism (VAL-CROSS-006, seed={args.seed}) ---")
    out1, err1, rc1 = run_once(args.model, prompt_ids, args.ngen, args.seed, False, args.threads)
    out2, err2, rc2 = run_once(args.model, prompt_ids, args.ngen, args.seed, False, args.threads)
    if rc1 != 0 or rc2 != 0:
        log(f"FAIL: engine exited {rc1}/{rc2}")
        sys.stderr.write(err1[-2000:] + "\n---\n" + err2[-2000:])
        all_passed = False
    elif out1 == out2:
        log(f"PASS (B): seeded runs byte-identical (sha {sha(out1)}, seed={args.seed})")
        # verify the seed appears in the engine's startup log
        if f"[seed] {args.seed}" in (err1 + err2):
            log(f"  startup log confirms seed={args.seed}")
        else:
            log(f"  WARN: seed={args.seed} not found in startup log")
    else:
        log(f"FAIL (B): seeded runs differ (sha1={sha(out1)} sha2={sha(out2)})")
        l1, l2 = out1.splitlines(), out2.splitlines()
        for i in range(min(len(l1), len(l2))):
            if l1[i] != l2[i]:
                log(f"  first divergence at line {i}: run1={l1[i]!r} run2={l2[i]!r}")
                break
        all_passed = False

    result_path = ROOT / "tests" / "oracle" / "determinism_test_result.json"
    result_path.write_text(
        f'{{"passed": {str(all_passed).lower()}, "seed": {args.seed}, '
        f'"greedy_sha": "{sha(out1)}", "prompt": "{args.prompt}"}}\n'
    )
    log(f"wrote {result_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
