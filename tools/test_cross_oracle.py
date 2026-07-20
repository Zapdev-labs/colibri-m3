#!/usr/bin/env python3
"""test_cross_oracle.py — f9: lightweight smoke test for cross_oracle_compare.py.

Validates the harness structure without requiring the 247GB GGUF load.
Checks:
  - cross_oracle_compare.py imports cleanly.
  - The colibri backend functions are present.
  - The llamacpp backend functions are present.
  - The result schema matches the documented shape.

Usage:
  python3 tools/test_cross_oracle.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[test_cross_oracle] {msg}", flush=True)


def main() -> int:
    rc = 0
    # 1. Import the harness.
    spec = importlib.util.spec_from_file_location(
        "cross_oracle_compare",
        str(ROOT / "tools" / "cross_oracle_compare.py"))
    if not spec or not spec.loader:
        log("FAIL: could not load cross_oracle_compare.py spec")
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log("OK: cross_oracle_compare.py imports cleanly")

    # 2. Check the key functions exist.
    for fn in ["colibri_greedy", "colibri_tf", "llamacpp_greedy", "llamacpp_tf",
               "tokenize_via_server", "main"]:
        if not hasattr(mod, fn):
            log(f"FAIL: missing function {fn}")
            rc = 1
        else:
            log(f"OK: {fn}() present")

    # 3. Check the contract constants.
    if mod.CONTRACT_FLOOR != 19:
        log(f"FAIL: CONTRACT_FLOOR should be 19 (VAL-CORR-021), got {mod.CONTRACT_FLOOR}")
        rc = 1
    else:
        log(f"OK: CONTRACT_FLOOR = {mod.CONTRACT_FLOOR}")

    # 4. Check the result path is under tests/oracle/.
    expected = ROOT / "tests" / "oracle" / "cross_oracle_result.json"
    if mod.RESULT_PATH != expected:
        log(f"FAIL: RESULT_PATH should be {expected}, got {mod.RESULT_PATH}")
        rc = 1
    else:
        log(f"OK: RESULT_PATH = {mod.RESULT_PATH}")

    # 5. If a cached result exists, validate its schema.
    if expected.is_file():
        try:
            result = json.loads(expected.read_text())
            for key in ["prompt", "prompt_ids", "colibri_model",
                        "llamacpp_model", "contract_floor"]:
                if key not in result:
                    log(f"FAIL: cached result missing key {key}")
                    rc = 1
            if "greedy" in result:
                g = result["greedy"]
                for key in ["colibri_ids", "llamacpp_ids", "matches", "passed"]:
                    if key not in g:
                        log(f"FAIL: cached greedy result missing key {key}")
                        rc = 1
                if "matches" in g and "n" in g:
                    log(f"OK: cached greedy result: {g['matches']}/{g['n']} match "
                        f"(passed={g.get('passed')})")
            log("OK: cached result schema is valid")
        except json.JSONDecodeError as e:
            log(f"FAIL: cached result is not valid JSON: {e}")
            rc = 1
    else:
        log("SKIP: no cached result (run cross_oracle_compare.py to produce one)")

    print(f"[test_cross_oracle] overall {'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
