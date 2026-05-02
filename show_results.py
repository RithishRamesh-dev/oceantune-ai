#!/usr/bin/env python3
"""
show_results.py  — print OceanTune benchmark results as a table or CSV.

Usage:
    python3 show_results.py                    # latest session, table view
    python3 show_results.py --csv              # CSV to stdout (pipe to file)
    python3 show_results.py --session <id>     # specific session
    python3 show_results.py --all              # all sessions
    python3 show_results.py --top 5            # top N by fitness
"""

import asyncio
import csv
import os
import sys
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient


MONGO_URI = os.environ.get("MONGO_URI", "")
DB_NAME   = os.environ.get("OCEANTUNE_DB", "oceantune")

COLUMNS = [
    ("session_id",          lambda r: r.get("session_id", "")[:8]),
    ("fingerprint",         lambda r: r.get("fingerprint", "")[:10]),
    ("fitness",             lambda r: f"{r.get('fitness_score', 0.0):.4f}"),
    ("ctx_in",              lambda r: r.get("context", {}).get("input_len", "")),
    ("ctx_out",             lambda r: r.get("context", {}).get("output_len", "")),
    ("tok/s",               lambda r: f"{r.get('enriched_metrics', r.get('raw_metrics', {})).get('peak_throughput_tokens_per_sec', 0.0):.1f}"),
    ("p95_ms",              lambda r: f"{r.get('enriched_metrics', r.get('raw_metrics', {})).get('p95_latency_at_peak_ms', 0.0):.1f}"),
    ("ttft_ms",             lambda r: f"{r.get('enriched_metrics', r.get('raw_metrics', {})).get('mean_ttft_ms', 0.0):.1f}"),
    ("valid_levels",        lambda r: r.get("enriched_metrics", {}).get("valid_levels", "")),
    ("error",               lambda r: (r.get("error") or "")[:40]),
]


async def fetch(session_id: Optional[str], top_n: int, all_sessions: bool):
    client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    # Resolve session
    if session_id:
        sid = session_id
    elif all_sessions:
        sid = None
    else:
        # Latest session by created_at
        doc = await db["sessions"].find_one(sort=[("created_at", -1)])
        if not doc:
            print("No sessions found in MongoDB.", file=sys.stderr)
            client.close()
            return []
        sid = str(doc["_id"])
        print(f"Latest session: {sid}  model={doc.get('model_id')}  gpu={doc.get('gpu_type')}\n",
              file=sys.stderr)

    query = {}
    if sid:
        query["session_id"] = sid

    cursor = db["benchmark_runs"].find(
        query,
        sort=[("fitness_score", -1)],
        limit=top_n,
    )
    rows = await cursor.to_list(length=top_n)
    client.close()
    return rows


def print_table(rows):
    headers = [c[0] for c in COLUMNS]
    data = [[fn(r) for _, fn in COLUMNS] for r in rows]

    widths = [max(len(h), max((len(str(d[i])) for d in data), default=0))
              for i, h in enumerate(headers)]

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in data:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)
    print(f"\n{len(rows)} row(s)")


def print_csv(rows):
    headers = [c[0] for c in COLUMNS]
    writer = csv.writer(sys.stdout)
    writer.writerow(headers)
    for r in rows:
        writer.writerow([fn(r) for _, fn in COLUMNS])


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--session", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()

    if not MONGO_URI:
        print("ERROR: MONGO_URI environment variable is not set.", file=sys.stderr)
        print("Set it with:  export MONGO_URI='mongodb+srv://...'", file=sys.stderr)
        sys.exit(1)

    rows = await fetch(
        session_id=args.session,
        top_n=args.top,
        all_sessions=args.all,
    )

    if not rows:
        print("No benchmark runs found.")
        return

    if args.csv:
        print_csv(rows)
    else:
        print_table(rows)


if __name__ == "__main__":
    asyncio.run(main())
