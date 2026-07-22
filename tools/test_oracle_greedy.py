#!/usr/bin/env python3
"""test_oracle_greedy.py — VAL-CORR-015: greedy decode matches oracle token IDs.

Runs colibri-m3 in greedy decode mode (TEMP=0) for 20 tokens and compares the
emitted token IDs against the cached oracle greedy artifact at
tests/oracle/greedy.json.

Per the f8-oracle-validation feature spec: the oracle was produced via the
llama.cpp GGUF Q4_K_M path. colibri-m3's per-row int4 quantization differs
from Q4_K_M block quantization, so an exact 20/20 match is the target but
>=18/20 is allowed for quantization drift. The validation contract
(VAL-CORR-021) tightens the cross-oracle floor to >=19/20; we report the
match count and exit 0 when >=18/20 (the feature-spec floor), flagging
whether the >=19/20 contract floor is also met.

Usage:
  python3 tools/test_oracle_greedy.py [--model /path/to/m3_i4]
                                      [--oracle tests/oracle/greedy.json]
                                      [--ngen 20]
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
DEFAULT_MODEL = os.environ.get("COLI_MODEL", "/path/to/m3_i4")
DEFAULT_ORACLE = ROOT / "tests" / "oracle" / "greedy.json"

NGEN_DEFAULT = 20
FLOOR = 18              # feature-spec floor: >=18/20 (int4-vs-Q4_K_M drift)
CONTRACT_FLOOR = 19     # VAL-CORR-021 tightened floor


def log(msg: str) -> None:
    print(f"[test_oracle_greedy] {msg}", file=sys.stderr, flush=True)


def run_engine_greedy(model: str, prompt_ids: list[int], ngen: int,
                      threads: int) -> list[int]:
    """Run the engine in greedy mode and return the emitted token IDs."""
    if not ENGIN.is_file():
        log(f"engine binary not found at {ENGIN} — run: make")
        sys.exit(1)
    env = os.environ.copy()
    env["SNAP"] = model
    env["TEMP"] = "0"           # greedy
    env["TOPP"] = "0"
    env["TOPK"] = "0"
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "close"
    cmd = [str(ENGIN), "128", "4", "8", "4096"]
    payload = f"{ngen}\n" + " ".join(map(str, prompt_ids)) + "\n"
    log(f"launching engine (greedy, prompt={len(prompt_ids)} toks, ngen={ngen})")
    proc = subprocess.run(cmd, input=payload, env=env, capture_output=True,
                          text=True, timeout=7200)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        sys.exit(f"engine exited {proc.returncode}")
    if proc.stderr:
        for line in proc.stderr.splitlines()[-8:]:
            log(f"  engine: {line}")
    ids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.isdigit() or (line.startswith("-") and line[1:].isdigit()):
            ids.append(int(line))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    ap.add_argument("--ngen", type=int, default=NGEN_DEFAULT)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "88")))
    args = ap.parse_args()

    if not Path(args.oracle).is_file():
        log(f"oracle artifact not found: {args.oracle}")
        log("run: python3 tools/make_m3_oracle.py --backend llamacpp")
        return 1
    oracle = json.loads(Path(args.oracle).read_text())
    prompt_ids = oracle["prompt_ids"]
    oracle_ids = oracle["greedy_ids"][:args.ngen]
    ngen = min(args.ngen, len(oracle_ids))
    log(f"oracle: prompt={len(prompt_ids)} ids, greedy={ngen} ids, backend={oracle.get('backend')}")

    colibri_ids = run_engine_greedy(args.model, prompt_ids, ngen, args.threads)
    if len(colibri_ids) < ngen:
        log(f"WARNING: engine emitted {len(colibri_ids)} tokens, expected {ngen}")

    n = min(len(colibri_ids), len(oracle_ids))
    matches = 0
    diffs = []
    for i in range(n):
        if colibri_ids[i] == oracle_ids[i]:
            matches += 1
        else:
            diffs.append((i, colibri_ids[i], oracle_ids[i]))

    meets_floor = matches >= FLOOR
    meets_contract = matches >= CONTRACT_FLOOR
    summary = (f"greedy match {matches}/{n} | floor {FLOOR} ({'PASS' if meets_floor else 'FAIL'}) | "
               f"contract {CONTRACT_FLOOR} ({'PASS' if meets_contract else 'FAIL'})")
    if meets_floor:
        summary = "PASS: " + summary
    else:
        summary = "FAIL: " + summary
    log(summary)
    for pos, c, o in diffs:
        log(f"  pos {pos}: colibri={c:6d} oracle={o:6d}")
    if colibri_ids[:n]:
        log(f"  colibri_ids={colibri_ids[:n]}")
        log(f"  oracle_ids ={oracle_ids[:n]}")

    result_path = ROOT / "tests" / "oracle" / "greedy_compare_result.json"
    result_path.write_text(json.dumps({
        "passed": meets_floor,
        "meets_contract_floor": meets_contract,
        "matches": matches,
        "n": n,
        "floor": FLOOR,
        "contract_floor": CONTRACT_FLOOR,
        "colibri_ids": colibri_ids[:n],
        "oracle_ids": oracle_ids[:n],
        "diffs": [{"pos": p, "colibri": c, "oracle": o} for p, c, o in diffs],
        "backend": oracle.get("backend"),
    }, indent=2))
    log(f"wrote {result_path}")
    return 0 if meets_floor else 1


if __name__ == "__main__":
    sys.exit(main())
