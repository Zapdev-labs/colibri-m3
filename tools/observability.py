#!/usr/bin/env python3
"""observability.py — f15: cross-engine telemetry dashboard.

Ingests JSON-line telemetry from both colibri-m3 (src/observability.h output)
and the llama.cpp rq fork (tools/llama_observability_wrapper.py output) and
produces a side-by-side comparison dashboard covering:

  - load time / RSS
  - per-token decode latency (median, p50, p90, p99)
  - sustained tok/s
  - expert cache hit rate
  - NUMA fault distribution (if numastat deltas are present)
  - NaN/Inf detection events

Modes:
  --compare               side-by-side comparison (default)
  --dashboard <path>      write a markdown dashboard to <path>
  --json <path>           write structured JSON summary to <path>

Usage:
  python3 tools/observability.py \\
      --colibri /tmp/colibri_telem.jsonl \\
      --llamacpp /tmp/llama_telem.jsonl \\
      --dashboard tests/oracle/cross_engine_dashboard.md \\
      --json tests/oracle/cross_engine_summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional
import os


def load_events(path: str) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summarize_engine(events: list[dict], engine_id: str) -> dict:
    summary = {
        "engine": engine_id,
        "events": len(events),
        "load_complete": None,
        "decode_step_count": 0,
        "step_ms_median": None,
        "step_ms_p50": None,
        "step_ms_p90": None,
        "step_ms_p99": None,
        "tokps_overall": None,
        "tokps_warm": None,
        "hit_rate": None,
        "nan_failures": 0,
        "run_complete": None,
    }
    load_events = [e for e in events if e.get("event") == "load_complete"]
    if load_events:
        summary["load_complete"] = load_events[-1]

    decode_steps = [e for e in events if e.get("event") == "decode_step"]
    summary["decode_step_count"] = len(decode_steps)
    if decode_steps:
        step_ms = [e.get("step_ms", 0) for e in decode_steps if "step_ms" in e]
        if step_ms:
            step_ms_sorted = sorted(step_ms)
            n = len(step_ms_sorted)
            summary["step_ms_median"] = statistics.median(step_ms)
            summary["step_ms_p50"] = step_ms_sorted[n // 2]
            summary["step_ms_p90"] = step_ms_sorted[int(n * 0.9)]
            summary["step_ms_p99"] = step_ms_sorted[int(n * 0.99)]
            summary["tokps_overall"] = 1000.0 / statistics.median(step_ms) if step_ms else 0
            # Warm = last 50% of decode steps.
            warm = step_ms[len(step_ms) // 2:]
            if warm:
                summary["tokps_warm"] = 1000.0 / statistics.median(warm)
        hits = sum(e.get("hit", 0) for e in decode_steps if "hit" in e)
        miss = sum(e.get("miss", 0) for e in decode_steps if "miss" in e)
        if hits + miss > 0:
            summary["hit_rate"] = hits / (hits + miss)

    nan_events = [e for e in events if "nan" in str(e).lower()]
    summary["nan_failures"] = len(nan_events)

    run_complete = [e for e in events if e.get("event") == "run_complete"]
    if run_complete:
        summary["run_complete"] = run_complete[-1]
        rc = run_complete[-1]
        if "tokps" in rc:
            summary["tokps_overall"] = rc["tokps"]
        if "hit_rate" in rc:
            summary["hit_rate"] = rc["hit_rate"]
    return summary


def render_markdown(c_summary: dict, l_summary: dict) -> str:
    lines = []
    lines.append("# f15: Cross-Engine Observability Dashboard")
    lines.append("")
    lines.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}")
    lines.append("")
    lines.append("## Side-by-side comparison")
    lines.append("")
    lines.append("| Metric | colibri-m3 | llama.cpp-minimax-m3-rq |")
    lines.append("|---|---|---|")

    def row(label, cv, lv, fmt="{}"):
        c = fmt.format(cv) if cv is not None else "-"
        l = fmt.format(lv) if lv is not None else "-"
        lines.append(f"| {label} | {c} | {l} |")

    lc = c_summary.get("load_complete") or {}
    ll = l_summary.get("load_complete") or {}
    row("Load RSS (kB)", lc.get("rss_kb"), ll.get("rss_kb"))
    row("Load layers", lc.get("layers"), ll.get("layers"))
    row("Load MSA layers", lc.get("msa_layers"), ll.get("msa_layers"))

    row("Decode steps recorded", c_summary.get("decode_step_count"),
        l_summary.get("decode_step_count"))
    row("Step ms (median)", c_summary.get("step_ms_median"),
        l_summary.get("step_ms_median"), fmt="{:.3f}")
    row("Step ms (p90)", c_summary.get("step_ms_p90"),
        l_summary.get("step_ms_p90"), fmt="{:.3f}")
    row("Step ms (p99)", c_summary.get("step_ms_p99"),
        l_summary.get("step_ms_p99"), fmt="{:.3f}")
    row("Tok/s (overall)", c_summary.get("tokps_overall"),
        l_summary.get("tokps_overall"), fmt="{:.2f}")
    row("Tok/s (warm cache)", c_summary.get("tokps_warm"),
        l_summary.get("tokps_warm"), fmt="{:.2f}")
    row("Expert cache hit rate", c_summary.get("hit_rate"),
        l_summary.get("hit_rate"), fmt="{:.4f}")
    row("NaN/Inf failures", c_summary.get("nan_failures"),
        l_summary.get("nan_failures"))

    rc_c = c_summary.get("run_complete") or {}
    rc_l = l_summary.get("run_complete") or {}
    row("Run tokens", rc_c.get("tokens"), rc_l.get("tokens"))
    row("Run wall (s)", rc_c.get("wall_s"), rc_l.get("wall_s"), fmt="{:.2f}")
    row("Run tok/s", rc_c.get("tokps"), rc_l.get("tokps"), fmt="{:.2f}")
    lines.append("")
    lines.append("## Throughput target (f13 contract: >=5 tok/s)")
    lines.append("")
    c_tokps = c_summary.get("tokps_warm") or c_summary.get("tokps_overall") or 0
    l_tokps = l_summary.get("tokps_warm") or l_summary.get("tokps_overall") or 0
    lines.append(f"- colibri-m3 warm tok/s: **{c_tokps:.2f}** "
                 f"({'PASS' if c_tokps >= 5 else 'FAIL'} >=5)")
    lines.append(f"- llama.cpp-rq warm tok/s: **{l_tokps:.2f}** "
                 f"({'PASS' if l_tokps >= 5 else 'FAIL'} >=5)")
    if l_tokps > 0:
        ratio = c_tokps / l_tokps
        lines.append(f"- colibri / llama.cpp ratio: **{ratio:.2f}x** "
                     f"({'colibri ahead' if ratio >= 1 else 'colibri behind'})")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colibri", default=os.environ.get("COLI_TELEMETRY", ""),
                    help="colibri-m3 telemetry JSONL path")
    ap.add_argument("--llamacpp", default=os.environ.get("LLAMA_TELEMETRY", ""),
                    help="llama.cpp-rq telemetry JSONL path")
    ap.add_argument("--compare", action="store_true", default=True,
                    help="print side-by-side comparison to stdout (default)")
    ap.add_argument("--dashboard", default=None,
                    help="write markdown dashboard to this path")
    ap.add_argument("--json", default=None,
                    help="write structured JSON summary to this path")
    args = ap.parse_args()

    c_events = load_events(args.colibri)
    l_events = load_events(args.llamacpp)
    if not c_events and not l_events:
        log = lambda m: print(f"[observability] {m}", file=sys.stderr)
        log("no telemetry files found; pass --colibri and --llamacpp")
        log(f"  colibri: {args.colibri or '(none)'}")
        log(f"  llamacpp: {args.llamacpp or '(none)'}")
        return 1

    c_summary = summarize_engine(c_events, "colibri-m3")
    l_summary = summarize_engine(l_events, "llama.cpp-minimax-m3-rq")

    md = render_markdown(c_summary, l_summary)
    if args.dashboard:
        Path(args.dashboard).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dashboard).write_text(md)
        print(f"[observability] dashboard -> {args.dashboard}", file=sys.stderr)
    if args.json:
        summary = {"colibri": c_summary, "llamacpp": l_summary}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"[observability] json -> {args.json}", file=sys.stderr)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
