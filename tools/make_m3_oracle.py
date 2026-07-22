#!/usr/bin/env python3
"""Make MiniMax M3 oracle reference artifacts.

Ports Colibri's ``c/tools/make_glm_oracle.py`` to MiniMax M3. Produces two cached
reference artifacts so downstream tests do not reload the ~852 GB BF16 model
(or the 247 GB Q4_K_M GGUF) on every run:

  - ``tests/oracle/logits.json`` : teacher-forcing per-token logits for 32 tokens
                                   given a fixed prompt.
  - ``tests/oracle/greedy.json``  : greedy-decode 20 token IDs via argmax.

Two backends are supported:

  - ``hf``       : HuggingFace ``transformers`` ``MiniMaxM3VLForCausalLM``
                   (the text-only CausalLM; the config's ``architectures`` field
                   ``MiniMaxM3SparseForCausalLM`` resolves to this class via
                   ``trust_remote_code=True``).  Loads BF16 weights; requires
                   ~852 GB on disk and enough RAM to materialise the active
                   layer's tensors (low_cpu_mem_usage + mmap streaming).
  - ``llamacpp`` : llama.cpp fork ``llama-server`` driving the 247 GB Q4_K_M GGUF
                   at ``/path/to/MiniMax-M3-MSA-GGUF/Q4_K_M/``.  Fits in
                   the 376 GB remote.  Used as the secondary oracle when BF16 is
                   unavailable (the documented fallback path).

The script is idempotent: if both artifacts already exist and ``--force`` is not
given, it prints the cached paths and exits 0 without reloading the model.

Usage:
    python tools/make_m3_oracle.py --prompt "The quick brown fox" --ngen 20 --backend llamacpp
    python tools/make_m3_oracle.py --prompt "Hello, my name is"  --ngen 20 --backend hf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults -- kept as module constants so tests can import them.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
ORACLE_DIR = REPO_ROOT / "tests" / "oracle"
LOGITS_PATH = ORACLE_DIR / "logits.json"
GREEDY_PATH = ORACLE_DIR / "greedy.json"
# Aliases matching VAL-CORR-013 naming (oracle_logits.json / oracle_greedy.json).
# Written alongside the feature-spec names so both contract and feature are satisfied.
LOGITS_ALIAS = ORACLE_DIR / "oracle_logits.json"
GREEDY_ALIAS = ORACLE_DIR / "oracle_greedy.json"

DEFAULT_PROMPT = "The quick brown fox"
DEFAULT_NGEN = 20            # greedy decode length
DEFAULT_TF_TOKENS = 32      # teacher-forcing length (>= ngen so greedy ⊂ tf)

# llama.cpp fork paths (READ-ONLY oracle -- never modified)
LLAMACPP_ROOT = Path("/path/to/llama.cpp-minimax-m3-rq")
LLAMACPP_BIN = LLAMACPP_ROOT / "build-simd" / "bin" / "llama-server"
GGUF_GLOB = "/path/to/MiniMax-M3-MSA-GGUF/Q4_K_M/*.gguf"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8399           # out-of-the-way of 3119 (mission HTTP stretch) and 8080 (llama default)

# HF paths
HF_META_DIR = "/path/to/m3-target-meta"   # config + tokenizer + auto_map
HFENV_PYTHON = "python3"         # transformers 5.12.1 + torch 2.12.1+cpu

VOCAB_SIZE = 200064          # from config.text_config.vocab_size


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[make_m3_oracle] {msg}", flush=True)


def artifacts_exist() -> bool:
    return LOGITS_PATH.is_file() and GREEDY_PATH.is_file()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # compact but readable: 2-space indent is fine for greedy (tiny);
    # logits can be large -- use compact separators there.
    if path.name == "logits.json":
        path.write_text(json.dumps(obj, separators=(",", ":")))
    else:
        path.write_text(json.dumps(obj, indent=2))
    log(f"wrote {path} ({path.stat().st_size:,} bytes)")


def write_aliases() -> None:
    """Also write oracle_logits.json / oracle_greedy.json (VAL-CORR-013 names)."""
    for src, dst in ((LOGITS_PATH, LOGITS_ALIAS), (GREEDY_PATH, GREEDY_ALIAS)):
        if src.is_file():
            dst.write_bytes(src.read_bytes())
            log(f"alias {dst} ({dst.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------

def _gguf_model_path() -> str:
    import glob
    shards = sorted(glob.glob(GGUF_GLOB))
    if not shards:
        raise FileNotFoundError(f"no GGUF shards at {GGUF_GLOB}")
    # llama-server accepts the first shard of a multi-part GGUF and loads the rest.
    return shards[0]


def _wait_for_server(url: str, timeout: int = 1800) -> None:
    """Poll /health until 200 or timeout."""
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
                if r.status == 200:
                    log("server is healthy")
                    return
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(5)
    raise RuntimeError(f"server did not become healthy within {timeout}s: {last_err}")


def _start_llamacpp_server(port: int, threads: int) -> subprocess.Popen:
    model = _gguf_model_path()
    log(f"starting llama-server: model={model} port={port} threads={threads}")
    cmd = [
        str(LLAMACPP_BIN),
        "-m", model,
        "--host", SERVER_HOST,
        "--port", str(port),
        "-t", str(threads),
        "-c", "4096",
        "-fa", "on",
        "-ngl", "0",          # CPU-only machine
        "--no-warmup",
        "--temp", "0",
        "--top-k", "1",       # greedy for default sampling; we still pass temp=0 per request
        "--top-p", "1.0",
        "--repeat-penalty", "1.0",
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LLAMACPP_ROOT}/build-simd/bin:{env.get('LD_LIBRARY_PATH','')}"
    # numactl --interleave=all per mission boundary
    cmd = ["numactl", "--interleave=all"] + cmd
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    log(f"server PID={proc.pid}")
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    log(f"stopping server PID={proc.pid}")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    log("server stopped")


def _server_post(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/completion", data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _server_tokenize(url: str, text: str) -> list[int]:
    payload = {"content": text, "add_special": True}
    req = urllib.request.Request(
        f"{url}/tokenize", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


def run_llamacpp_backend(prompt: str, ngen: int, tf_tokens: int, port: int,
                         threads: int, keep_server: bool) -> None:
    """Use llama-server to produce both artifacts in one model load.

    Strategy: generate ``tf_tokens`` tokens greedily with a large ``n_probs``
    so the response carries per-position token probabilities. Greedy
    generation IS teacher-forcing when the targets are the argmax
    continuation: the logits computed at each step are the
    teacher-forcing logits for the (prompt + previous tokens) prefix.
    """
    url = f"http://{SERVER_HOST}:{port}"
    proc = None
    started_here = False

    # Reuse a running server if one is already up on the port.
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
            if r.status == 200:
                log(f"reusing running server at {url}")
    except Exception:
        proc = _start_llamacpp_server(port, threads)
        started_here = True
        _wait_for_server(url)

    try:
        prompt_ids = _server_tokenize(url, prompt)
        log(f"prompt tokenised to {len(prompt_ids)} ids: {prompt_ids[:12]}{'...' if len(prompt_ids)>12 else ''}")

        # One completion call: greedy + per-position top_probs.
        # n_probs caps the per-position returned top-K. We request as many as
        # possible; the server clamps if it must. Full vocab (200064) is ~6 MB
        # of JSON per position, ~200 MB total -- too large to be practical, so
        # default to a large-but-bounded K and also save the argmax IDs.
        n_probs = int(os.environ.get("M3_ORACLE_NPROBS", "200"))
        n_predict = max(ngen, tf_tokens)
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "seed": 424242,
            "stream": False,
            "n_probs": n_probs,
            "cache_prompt": False,
        }
        log(f"POST /completion n_predict={n_predict} n_probs={n_probs} ...")
        t0 = time.time()
        resp = _server_post(url, payload, timeout=3600)
        log(f"completion returned in {time.time()-t0:.1f}s")

        # Parse response
        content = resp.get("content", "")
        # /completion returns `completion_probabilities` with per-step
        # {id, token, bytes, logprob, top_logprobs:[{id,token,bytes,logprob},...]}
        # Each step's logits are computed by feeding prompt+previous tokens,
        # so this IS teacher-forcing when targets are the greedy continuation.
        comps = resp.get("completion_probabilities") or []
        if not comps:
            raise RuntimeError("server returned no completion_probabilities; cannot build logits artifact")

        gen_ids: list[int] = []
        per_pos_logits: list[dict] = []
        for step in comps:
            top = step.get("top_logprobs") or []
            if not top:
                continue
            # top is already sorted by logprob desc; top[0] is the argmax.
            argmax_id = int(top[0]["id"])
            gen_ids.append(argmax_id)
            # Save [token_id, logit=logprob] per step (top-K, capped by n_probs).
            step_logits = [[int(e["id"]), float(e["logprob"])] for e in top]
            per_pos_logits.append({
                "argmax": argmax_id,
                "argmax_logprob": float(step.get("logprob", top[0]["logprob"])),
                "top": step_logits,
            })

        if len(gen_ids) < n_predict:
            log(f"WARNING: only got {len(gen_ids)}/{n_predict} generation steps")

        greedy_ids = gen_ids[:ngen]
        tf_ids = gen_ids[:tf_tokens]

        greedy_obj = {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "greedy_ids": greedy_ids,
            "n_gen": len(greedy_ids),
            "backend": "llamacpp",
            "model": "MiniMax-M3-Q4_K_M-GGUF",
            "seed": 424242,
        }

        logits_obj = {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "tf_target_ids": tf_ids,
            "n_tf": len(tf_ids),
            "vocab_size": VOCAB_SIZE,
            "backend": "llamacpp",
            "model": "MiniMax-M3-Q4_K_M-GGUF",
            "seed": 424242,
            "n_probs_per_step": n_probs,
            "note": ("Teacher-forcing via greedy continuation: each step's "
                     "logits are computed by feeding prompt+previous tokens. "
                     "Per-step top-K token probs are stored (prob->logit=log). "
                     "Full-vocab BF16 logits require the hf backend."),
            "per_position": per_pos_logits[:tf_tokens],
        }

        write_json(GREEDY_PATH, greedy_obj)
        write_json(LOGITS_PATH, logits_obj)
        write_aliases()
        log(f"greedy_ids={greedy_ids}")
        log(f"tf_target_ids[:12]={tf_ids[:12]}")

    finally:
        if started_here and not keep_server:
            _stop_server(proc)
        elif started_here and keep_server:
            log(f"server left running (PID={proc.pid}) for reuse; "
                f"stop with: kill {proc.pid}")


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

def run_hf_backend(prompt: str, ngen: int, tf_tokens: int,
                   model_path: str | None) -> None:
    """Load HF MiniMaxM3VLForCausalLM and produce both artifacts.

    Requires the BF16 checkpoint on disk. See README / mission notes for
    RAM constraints (852 GB BF16 -- use low_cpu_mem_usage + mmap streaming).

    NOTE: ``MiniMaxM3SparseForCausalLM`` (the name in the config's
    ``text_config.architectures``) is not a real transformers class. The
    text-only CausalLM that implements it is ``MiniMaxM3VLForCausalLM``
    (see ``transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py``).
    We import it explicitly rather than relying on AutoModelForCausalLM,
    because the top-level config is a VL conditional-generation config
    (``MiniMaxM3SparseForConditionalGeneration``) and AutoModelForCausalLM
    would try to resolve against the top-level architectures field.
    """
    import torch  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (  # type: ignore
        MiniMaxM3VLForCausalLM,
    )

    model_path = model_path or HF_META_DIR
    log(f"loading HF model from {model_path} (trust_remote_code=True)...")
    log("NOTE: this requires the full BF16 checkpoint (~852 GB) on disk and "
        "enough RAM to stream the active layer. If the weights are absent this "
        "will fail with FileNotFoundError or OSError.")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = MiniMaxM3VLForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    log("model loaded")

    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    import torch as _torch
    inp = _torch.tensor([prompt_ids])

    # Greedy decode (ngen tokens)
    with _torch.no_grad():
        out = model.generate(inp, max_new_tokens=ngen, do_sample=False,
                             use_cache=True, temperature=None, top_k=None, top_p=None)
    full_ids = out[0].tolist()
    greedy_ids = full_ids[len(prompt_ids):len(prompt_ids) + ngen]
    log(f"greedy_ids={greedy_ids}")

    # Teacher-forcing: single forward pass over prompt + tf_tokens target tokens.
    tf_target = full_ids[len(prompt_ids):len(prompt_ids) + tf_tokens]
    tf_input = _torch.tensor([prompt_ids + tf_target])
    with _torch.no_grad():
        logits = model(tf_input, use_cache=False).logits[0]  # [seq, vocab]
    # Take logits at the positions that PREDICT each tf_target token.
    # Position i predicts token i+1; the tf_target starts right after the prompt.
    tf_start = len(prompt_ids) - 1
    tf_logits = logits[tf_start:tf_start + tf_tokens].tolist()  # [tf_tokens, vocab]

    greedy_obj = {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "greedy_ids": greedy_ids,
        "n_gen": len(greedy_ids),
        "backend": "hf",
        "model": "MiniMaxM3VLForCausalLM",
        "torch_dtype": "bfloat16",
    }
    logits_obj = {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "tf_target_ids": tf_target,
        "n_tf": len(tf_target),
        "vocab_size": logits.shape[-1],
        "backend": "hf",
        "model": "MiniMaxM3VLForCausalLM",
        "torch_dtype": "bfloat16",
        "logits": tf_logits,
    }
    write_json(GREEDY_PATH, greedy_obj)
    write_json(LOGITS_PATH, logits_obj)
    write_aliases()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--ngen", type=int, default=DEFAULT_NGEN,
                    help="greedy decode length (default 20)")
    ap.add_argument("--tf-tokens", type=int, default=DEFAULT_TF_TOKENS,
                    help="teacher-forcing length (default 32, must be >= ngen)")
    ap.add_argument("--backend", choices=["hf", "llamacpp", "auto"], default="auto",
                    help="oracle backend (auto: try hf, fall back to llamacpp)")
    ap.add_argument("--model", default=None,
                    help="HF model path (hf backend only; default: m3-target-meta)")
    ap.add_argument("--port", type=int, default=SERVER_PORT,
                    help="llama-server port (llamacpp backend)")
    ap.add_argument("--threads", type=int, default=88,
                    help="llama-server threads (llamacpp backend; default 88)")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if cached artifacts exist")
    ap.add_argument("--keep-server", action="store_true",
                    help="llamacpp: leave the server running for reuse by later steps")
    args = ap.parse_args()

    if args.tf_tokens < args.ngen:
        ap.error("--tf-tokens must be >= --ngen (greedy is a prefix of teacher-forcing)")

    if not args.force and artifacts_exist():
        log(f"cached artifacts already present:")
        log(f"  logits: {LOGITS_PATH}")
        log(f"  greedy: {GREEDY_PATH}")
        log("use --force to regenerate.")
        # Still verify shapes even when skipping.
        verify_artifacts(args.ngen, args.tf_tokens)
        return 0

    backend = args.backend
    if backend == "auto":
        # Prefer HF if the BF16 checkpoint is present; otherwise fall back.
        hf_ready = _hf_checkpoint_present(args.model)
        backend = "hf" if hf_ready else "llamacpp"
        log(f"auto-selected backend: {backend}")

    if backend == "hf":
        try:
            run_hf_backend(args.prompt, args.ngen, args.tf_tokens, args.model)
        except Exception as e:  # noqa: BLE001
            log(f"HF backend failed: {e}")
            log("falling back to llamacpp backend")
            run_llamacpp_backend(args.prompt, args.ngen, args.tf_tokens,
                                 args.port, args.threads, args.keep_server)
    else:
        run_llamacpp_backend(args.prompt, args.ngen, args.tf_tokens,
                             args.port, args.threads, args.keep_server)

    # Verify
    if not artifacts_exist():
        log("ERROR: artifacts not written")
        return 1
    if not verify_artifacts(args.ngen, args.tf_tokens):
        log("ERROR: artifact verification failed")
        return 1
    log("done.")
    return 0


def verify_artifacts(ngen: int, tf_tokens: int) -> bool:
    """Check cached artifacts have the expected structure/shapes."""
    ok = True
    try:
        g = json.loads(GREEDY_PATH.read_text())
        n = len(g.get("greedy_ids", []))
        if n != ngen:
            log(f"  greedy.json: expected {ngen} ids, got {n}")
            ok = False
        else:
            log(f"  greedy.json: OK ({n} greedy ids, backend={g.get('backend')})")
    except Exception as e:  # noqa: BLE001
        log(f"  greedy.json: FAIL {e}")
        ok = False
    try:
        l = json.loads(LOGITS_PATH.read_text())
        npos = len(l.get("per_position", []))
        if npos != tf_tokens:
            log(f"  logits.json: expected {tf_tokens} positions, got {npos}")
            ok = False
        else:
            log(f"  logits.json: OK ({npos} positions, "
                f"top-K={len(l['per_position'][0]['top'])}, "
                f"backend={l.get('backend')})")
    except Exception as e:  # noqa: BLE001
        log(f"  logits.json: FAIL {e}")
        ok = False
    return ok


def _hf_checkpoint_present(model_path: str | None) -> bool:
    """Heuristic: does the HF model dir contain weight shards?"""
    import glob
    p = model_path or HF_META_DIR
    return bool(glob.glob(os.path.join(p, "*.safetensors"))
                or glob.glob(os.path.join(p, "*.bin"))
                or glob.glob(os.path.join(p, "model-*")))


if __name__ == "__main__":
    sys.exit(main())
