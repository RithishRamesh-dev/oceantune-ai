#!/usr/bin/env python3
"""
show_results.py  — view OceanTune benchmark results as a table or CSV.

Usage
-----
  python3 show_results.py                          # latest session summary
  python3 show_results.py --levels                 # include per-concurrency breakdown
  python3 show_results.py --csv                    # CSV to stdout
  python3 show_results.py --session <id>           # specific session
  python3 show_results.py --all                    # all sessions
  python3 show_results.py --top 10                 # top N by fitness

Environment
-----------
  MONGO_URI   MongoDB connection string (required)
  OCEANTUNE_DB  database name (default: oceantune)
"""

import asyncio
import csv
import os
import sys
from dataclasses import fields
from typing import List, Optional

from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.environ.get("MONGO_URI", "")
DB_NAME   = os.environ.get("OCEANTUNE_DB", "oceantune")

# ── Summary columns (one row per benchmark run) ────────────────────────────────
SUMMARY_COLS = [
    ("session",    lambda r: r.get("session_id", "")[:8]),
    ("iter",       lambda r: _flag(r, "run_id", "")),  # not stored; use fingerprint order
    ("fingerprint",lambda r: r.get("fingerprint", "")[:12]),
    ("ctx_in",     lambda r: r.get("context", {}).get("input_len", "")),
    ("ctx_out",    lambda r: r.get("context", {}).get("output_len", "")),
    ("fitness",    lambda r: f"{r.get('fitness_score', 0.0):.4f}"),
    ("tok/s",      lambda r: _fmt(_em(r, "peak_throughput_tokens_per_sec"), ".1f")),
    ("req/s",      lambda r: _fmt(_em(r, "peak_requests_per_sec"), ".2f")),
    ("p95_ms",     lambda r: _fmt(_em(r, "p95_latency_at_peak_ms"), ".1f")),
    ("ttft_ms",    lambda r: _fmt(_em(r, "mean_ttft_ms"), ".1f")),
    ("tpot_ms",    lambda r: _fmt(_em(r, "mean_tpot_ms"), ".2f")),
    ("err_rate",   lambda r: _fmt(_em(r, "error_rate_max"), ".3f")),
    ("ok_levels",  lambda r: str(_em(r, "valid_levels") or "")),
    ("best_c",     lambda r: str(_em(r, "best_concurrency") or "")),
    ("gpu_util",   lambda r: _flag(r, "gpu_memory_utilization", "")),
    ("prefix_cache",lambda r: str(_flag(r, "enable_prefix_caching", ""))),
    ("block_size", lambda r: _flag(r, "block_size", "")),
    ("max_seqs",   lambda r: _flag(r, "max_num_seqs", "")),
    ("batched_toks",lambda r: _flag(r, "max_num_batched_tokens", "")),
    ("error",      lambda r: (r.get("error") or "")[:50]),
]

# ── Per-level breakdown columns ────────────────────────────────────────────────
LEVEL_COLS = [
    ("fingerprint",lambda r, lv: r.get("fingerprint", "")[:12]),
    ("ctx_in",     lambda r, lv: r.get("context", {}).get("input_len", "")),
    ("ctx_out",    lambda r, lv: r.get("context", {}).get("output_len", "")),
    ("concurrency",lambda r, lv: lv.get("concurrency", "")),
    ("n_success",  lambda r, lv: lv.get("num_prompts", "")),
    ("tok/s",      lambda r, lv: _fmt(lv.get("output_tokens_per_sec"), ".1f")),
    ("req/s",      lambda r, lv: _fmt(lv.get("requests_per_sec"), ".2f")),
    ("p50_ms",     lambda r, lv: _fmt(lv.get("median_latency_ms"), ".1f")),
    ("p95_ms",     lambda r, lv: _fmt(lv.get("p95_latency_ms"), ".1f")),
    ("p99_ms",     lambda r, lv: _fmt(lv.get("p99_latency_ms"), ".1f")),
    ("ttft_ms",    lambda r, lv: _fmt(lv.get("mean_ttft_ms"), ".1f")),
    ("tpot_ms",    lambda r, lv: _fmt(lv.get("mean_tpot_ms"), ".2f")),
    ("err_rate",   lambda r, lv: _fmt(lv.get("error_rate"), ".3f")),
    ("failed",     lambda r, lv: str(lv.get("failed", ""))),
    ("dur_s",      lambda r, lv: _fmt(lv.get("duration_sec"), ".1f")),
]


def _em(r: dict, key: str):
    """Read from enriched_metrics (preferred) then raw_metrics."""
    v = (r.get("enriched_metrics") or {}).get(key)
    if v is None:
        v = (r.get("raw_metrics") or {}).get(key)
    return v


def _flag(r: dict, key: str, default=""):
    return r.get("flags", {}).get(key, default)


def _fmt(v, fmt: str):
    if v is None or v == "":
        return "—"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


async def fetch(
    session_id: Optional[str],
    top_n: int,
    all_sessions: bool,
):
    if not MONGO_URI:
        print("ERROR: MONGO_URI is not set.\n  export MONGO_URI='mongodb+srv://...'",
              file=sys.stderr)
        sys.exit(1)

    client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    if session_id:
        sid = session_id
        doc = await db["sessions"].find_one({"_id": __import__("bson").ObjectId(sid)})
        label = f"session={sid[:8]}  model={doc.get('model_id') if doc else '?'}"
    elif all_sessions:
        sid = None
        label = "all sessions"
    else:
        doc = await db["sessions"].find_one(sort=[("created_at", -1)])
        if not doc:
            print("No sessions in MongoDB.", file=sys.stderr)
            client.close()
            return [], ""
        sid = str(doc["_id"])
        label = (f"Latest session: {sid}  "
                 f"model={doc.get('model_id')}  gpu={doc.get('gpu_type')}  "
                 f"status={doc.get('status')}")

    query = {"session_id": sid} if sid else {}
    cursor = (
        db["benchmark_runs"]
        .find(query)
        .sort("fitness_score", -1)
        .limit(top_n)
    )
    rows = await cursor.to_list(length=top_n)
    client.close()
    return rows, label


def _table(headers, data):
    if not data:
        return "(no rows)"
    widths = [max(len(h), max((len(str(row[i])) for row in data), default=0))
              for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    lines = [sep, fmt.format(*headers), sep]
    for row in data:
        lines.append(fmt.format(*[str(v) for v in row]))
    lines.append(sep)
    return "\n".join(lines)


def print_summary(rows, as_csv: bool):
    headers = [c[0] for c in SUMMARY_COLS]
    data = [[fn(r) for _, fn in SUMMARY_COLS] for r in rows]
    if as_csv:
        w = csv.writer(sys.stdout)
        w.writerow(headers)
        for row in data:
            w.writerow(row)
    else:
        print(_table(headers, data))
        print(f"\n{len(rows)} run(s)")


def print_levels(rows, as_csv: bool):
    headers = [c[0] for c in LEVEL_COLS]
    data = []
    for r in rows:
        for lv in r.get("levels", []):
            data.append([fn(r, lv) for _, fn in LEVEL_COLS])
    if not data:
        print("No per-level data found. Re-run the pipeline to populate levels.")
        return
    if as_csv:
        w = csv.writer(sys.stdout)
        w.writerow(headers)
        for row in data:
            w.writerow(row)
    else:
        print(_table(headers, data))
        print(f"\n{len(data)} level(s) across {len(rows)} run(s)")


async def main():
    import argparse
    p = argparse.ArgumentParser(description="View OceanTune benchmark results")
    p.add_argument("--session", default=None, help="Session ID (default: latest)")
    p.add_argument("--all", action="store_true", help="Show all sessions")
    p.add_argument("--top", type=int, default=50, help="Max rows (default 50)")
    p.add_argument("--levels", action="store_true",
                   help="Show per-concurrency-level breakdown instead of summary")
    p.add_argument("--csv", action="store_true", help="CSV output")
    args = p.parse_args()

    rows, label = await fetch(
        session_id=args.session,
        top_n=args.top,
        all_sessions=args.all,
    )
    if not rows:
        print("No benchmark runs found.")
        return

    if not args.csv:
        print(f"\n{label}\n")

    if args.levels:
        print_levels(rows, as_csv=args.csv)
    else:
        print_summary(rows, as_csv=args.csv)


if __name__ == "__main__":
    asyncio.run(main())
