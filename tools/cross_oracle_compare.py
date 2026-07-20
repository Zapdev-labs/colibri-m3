#!/usr/bin/env python3
"""cross_oracle_compare.py — f9: VAL-CORR-021 cross-engine agreement.

Drives both engines on identical prompts + token IDs and reports greedy
token-ID agreement between colibri-m3 and the llama.cpp-minimax-m3-rq fork
(the independent Q4_K_M oracle). The mission's correctness gate is >=19/20
token IDs matching across two independent int4 quantization schemes.

Modes:
  --mode greedy        greedy-decode N tokens on both engines, compare IDs.
  --mode tf            teacher-force the same N target tokens on both, compare
                       per-position argmax + top-K overlap.
  --mode both          greedy + tf.

This harness does NOT reload the 247GB model on every call: the llama.cpp
backend is driven via a long-running llama-server (started once, reused across
prompts) so the per-prompt cost is ~seconds, not minutes.

Usage:
  python3 tools/cross_oracle_compare.py --mode both --ngen 20 --tf-tokens 32 \
      --prompt 'The quick brown fox jumps over the lazy dog'

Outputs JSON to tests/oracle/cross_oracle_result.json with the same shape as
greedy_compare_result.json plus a per-mode agreement breakdown.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGIN = ROOT / "m3"
RESULT_PATH = ROOT / "tests" / "oracle" / "cross_oracle_result.json"

DEFAULT_MODEL = os.environ.get("COLI_MODEL", "/home/ai/models/m3_i4_v3")
LLAMACPP_BIN = Path("/home/ai/llama.cpp-minimax-m3-rq/build-simd/bin/llama-server")
GGUF_GLOB = "/home/ai/models/MiniMax-M3-MSA-GGUF/Q4_K_M/*.gguf"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = int(os.environ.get("CROSS_PORT", "8399"))

CONTRACT_FLOOR = 19   # VAL-CORR-021: >=19/20 cross-oracle greedy agreement
FLOOR = 18
TF_ARGMAX_FLOOR = 30  # 30/32 cross-oracle TF argmax agreement


def log(m: str) -> None:
    print(f"[cross_oracle] {m}", flush=True)


def tokenize_via_server(url: str, text: str) -> list[int]:
    payload = {"content": text, "add_special": True}
    req = urllib.request.Request(
        f"{url}/tokenize", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


def llamacpp_greedy(url: str, prompt_ids: list[int], ngen: int) -> list[int]:
    """Drive llama-server greedy: feed prompt IDs as a list, get back token IDs."""
    payload = {
        "prompt": prompt_ids, "n_predict": ngen, "temperature": 0.0,
        "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
        "seed": 424242, "stream": False, "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"{url}/completion", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.loads(r.read())
    # Parse the /completion response: tokens field is a list of ints.
    tokens = resp.get("tokens") or []
    if not tokens:
        # Some versions return content string + completion_probs; fall back.
        content = resp.get("content", "")
        # Re-tokenize the content to recover IDs (less precise but works).
        tokens = tokenize_via_server(url, content)
    return [int(t) for t in tokens][:ngen]


def colibri_greedy(model: str, prompt_ids: list[int], ngen: int, threads: int) -> list[int]:
    if not ENGIN.is_file():
        sys.exit(f"engine binary missing at {ENGIN} — run: make")
    env = os.environ.copy()
    env["SNAP"] = model
    env["TEMP"] = "0"; env["TOPP"] = "0"; env["TOPK"] = "0"
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PLACES"] = "cores"; env["OMP_PROC_BIND"] = "close"
    env["SEED"] = "424242"
    payload = f"{ngen}\n" + " ".join(map(str, prompt_ids)) + "\n"
    log(f"launching colibri-m3 (prompt={len(prompt_ids)} toks, ngen={ngen})")
    proc = subprocess.run([str(ENGIN), "128", "4", "8", "4096"],
                          input=payload, env=env, capture_output=True,
                          text=True, timeout=7200)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        sys.exit(f"colibri engine exited {proc.returncode}")
    ids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.isdigit() or (line.startswith("-") and line[1:].isdigit()):
            ids.append(int(line))
    return ids[:ngen]


def colibri_tf(model: str, prompt_ids: list[int], targets: list[int],
                topk: int, threads: int) -> list[dict]:
    """Run colibri teacher-force mode, return per-position dicts."""
    env = os.environ.copy()
    env["SNAP"] = model
    env["TF_MODE"] = "1"; env["TF_TOPK"] = str(topk)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PLACES"] = "cores"; env["OMP_PROC_BIND"] = "close"
    import tempfile
    tf_out = tempfile.NamedTemporaryFile(mode="w", suffix=".tf", delete=False)
    tf_out.close()
    env["TF_OUT"] = tf_out.name
    payload = f"{len(targets)}\n" + " ".join(map(str, prompt_ids)) + "\n" \
              + " ".join(map(str, targets)) + "\n"
    log(f"launching colibri-m3 TF (prompt={len(prompt_ids)} toks, tf={len(targets)})")
    proc = subprocess.run([str(ENGIN), "--teacher-force", "128", "4", "8", "4096"],
                          input=payload, env=env, capture_output=True,
                          text=True, timeout=7200)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        sys.exit(f"colibri TF exited {proc.returncode}")
    try:
        text = open(tf_out.name).read()
    finally:
        os.unlink(tf_out.name)
    positions = []
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("TF_START") or line.startswith("TF_END"):
            continue
        if line.startswith("POS "):
            if cur is not None:
                positions.append(cur)
            parts = line.split()
            cur = {"argmax": int(parts[2].split("=")[1]),
                   "argmax_lp": float(parts[3].split("=")[1]),
                   "top": []}
        elif cur is not None:
            tid, lp = line.split()
            cur["top"].append([int(tid), float(lp)])
    if cur is not None:
        positions.append(cur)
    return positions


def llamacpp_tf(url: str, prompt_ids: list[int], targets: list[int],
                n_probs: int) -> list[dict]:
    """Drive llama-server in TF mode by feeding prompt+prev_targets greedily."""
    # One completion call with n_predict = len(targets); the server returns
    # completion_probabilities per step. Greedy continuation IS teacher-forcing
    # when targets are the argmax continuation. We supply the oracle target IDs
    # so the comparison matches the same prefixes.
    # NOTE: llama-server /completion does not accept arbitrary target token
    # sequences to force-feed; it always samples. For greedy targets (which
    # are the argmax continuation), greedy n_predict == TF on the same prefix.
    # For non-greedy targets we'd need the embeddings API (out of scope).
    payload = {
        "prompt": prompt_ids, "n_predict": len(targets),
        "temperature": 0.0, "top_k": 1, "top_p": 1.0,
        "repeat_penalty": 1.0, "seed": 424242,
        "stream": False, "n_probs": n_probs, "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"{url}/completion", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.loads(r.read())
    comps = resp.get("completion_probabilities") or []
    out = []
    for step in comps:
        top = step.get("top_logprobs") or []
        out.append({
            "argmax": int(top[0]["id"]) if top else -1,
            "argmax_lp": float(step.get("logprob", top[0]["logprob"] if top else 0)),
            "top": [[int(e["id"]), float(e["logprob"])] for e in top],
        })
    return out[:len(targets)]


def wait_for_server(url: str, timeout: int = 1800) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"server not healthy within {timeout}s")


def start_server(port: int, threads: int) -> subprocess.Popen:
    import glob
    shards = sorted(glob.glob(GGUF_GLOB))
    if not shards:
        sys.exit(f"no GGUF shards at {GGUF_GLOB}")
    cmd = ["numactl", "--interleave=all", str(LLAMACPP_BIN),
           "-m", shards[0], "--host", SERVER_HOST, "--port", str(port),
           "-t", str(threads), "-c", "4096", "-fa", "on", "-ngl", "0",
           "--no-warmup", "--temp", "0", "--top-k", "1",
           "--top-p", "1.0", "--repeat-penalty", "1.0"]
    env = os.environ.copy()
    ld = f"/home/ai/llama.cpp-minimax-m3-rq/build-simd/bin:{env.get('LD_LIBRARY_PATH','')}"
    env["LD_LIBRARY_PATH"] = ld
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    log(f"started llama-server PID={proc.pid} port={port}")
    wait_for_server(f"http://{SERVER_HOST}:{port}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception: pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("--ngen", type=int, default=20)
    ap.add_argument("--tf-tokens", type=int, default=32)
    ap.add_argument("--mode", choices=["greedy", "tf", "both"], default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=88)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--keep-server", action="store_true",
                    help="don't kill llama-server on exit (reuse across runs)")
    ap.add_argument("--port", type=int, default=SERVER_PORT)
    args = ap.parse_args()

    url = f"http://{SERVER_HOST}:{args.port}"
    proc = None
    started_here = False
    try:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                if r.status == 200:
                    log(f"reusing running llama-server at {url}")
        except Exception:
            proc = start_server(args.port, args.threads)
            started_here = True

        prompt_ids = tokenize_via_server(url, args.prompt)
        log(f"prompt tokenized: {len(prompt_ids)} ids: {prompt_ids[:12]}")

        result = {"prompt": args.prompt, "prompt_ids": prompt_ids,
                  "colibri_model": args.model,
                  "llamacpp_model": "MiniMax-M3-Q4_K_M-GGUF",
                  "contract_floor": CONTRACT_FLOOR,
                  "floor": FLOOR,
                  "mode": args.mode}

        if args.mode in ("greedy", "both"):
            c_ids = colibri_greedy(args.model, prompt_ids, args.ngen, args.threads)
            l_ids = llamacpp_greedy(url, prompt_ids, args.ngen)
            matches = sum(1 for a, b in zip(c_ids, l_ids) if a == b)
            diffs = [{"pos": i, "colibri": c_ids[i], "llamacpp": l_ids[i]}
                     for i in range(min(len(c_ids), len(l_ids))) if c_ids[i] != l_ids[i]]
            result["greedy"] = {
                "colibri_ids": c_ids, "llamacpp_ids": l_ids,
                "matches": matches, "n": len(c_ids),
                "passed": matches >= CONTRACT_FLOOR,
                "meets_contract_floor": matches >= CONTRACT_FLOOR,
                "meets_floor": matches >= FLOOR,
                "diffs": diffs,
            }
            log(f"greedy: {matches}/{len(c_ids)} match (contract={CONTRACT_FLOOR})")
            log(f"  colibri: {c_ids}")
            log(f"  llamacpp: {l_ids}")

        if args.mode in ("tf", "both"):
            # TF: feed oracle target IDs (from greedy continuation) to both engines.
            # For colibri we use --teacher-force mode; for llamacpp we drive greedy
            # continuation with n_predict=len(tf_tokens), which is TF when targets
            # are the argmax continuation.
            target_ids = llamacpp_greedy(url, prompt_ids, args.tf_tokens)
            c_tf = colibri_tf(args.model, prompt_ids, target_ids, args.topk, args.threads)
            l_tf = llamacpp_tf(url, prompt_ids, target_ids, args.topk)
            argmax_match = sum(1 for c, l in zip(c_tf, l_tf) if c["argmax"] == l["argmax"])
            # Per-position top-K overlap (Jaccard of token ID sets).
            overlaps = []
            for c, l in zip(c_tf, l_tf):
                cs = {t[0] for t in c["top"]}
                ls = {t[0] for t in l["top"]}
                u = cs | ls
                overlaps.append(len(cs & ls) / len(u) if u else 1.0)
            mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
            result["tf"] = {
                "target_ids": target_ids,
                "colibri_argmax": [c["argmax"] for c in c_tf],
                "llamacpp_argmax": [l["argmax"] for l in l_tf],
                "argmax_matches": argmax_match,
                "n_positions": len(c_tf),
                "argmax_floor": TF_ARGMAX_FLOOR,
                "passed": argmax_match >= TF_ARGMAX_FLOOR,
                "mean_topk_overlap": mean_overlap,
                "per_pos_overlap": overlaps,
            }
            log(f"tf: {argmax_match}/{len(c_tf)} argmax match (floor={TF_ARGMAX_FLOOR}); "
                f"mean top-K overlap {mean_overlap*100:.1f}%")

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2))
        log(f"wrote {RESULT_PATH}")

        passed = ("greedy" not in result or result["greedy"]["passed"]) and \
                 ("tf" not in result or result["tf"]["passed"])
        return 0 if passed else 1
    finally:
        if started_here and not args.keep_server:
            stop_server(proc)
        elif started_here and args.keep_server and proc:
            log(f"server left running PID={proc.pid}; kill with: kill {proc.pid}")


if __name__ == "__main__":
    sys.exit(main())
