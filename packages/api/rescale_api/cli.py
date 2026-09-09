"""`uv run rescale <command>`."""
from __future__ import annotations

import argparse

from rescale_api import pipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="rescale", description="Synced .lrc lyrics for your music library.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="read-only library walk, upsert tracks into the DB")
    s.add_argument("--report", action="store_true", help="only print what the scanner sees, write nothing")
    r = sub.add_parser("run", help="process tracks (pending by default) and write .lrc sidecars")
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--redo-review", action="store_true", help="also re-run tracks in needs-review")
    r.add_argument("--filter", help="only tracks whose path contains this substring")
    r.add_argument("--limit", type=int)
    r.add_argument("--embed", action="store_true", default=None,
                    help="embed lyrics into an ID3 USLT frame instead of writing a .lrc sidecar (default: config [writer] embed)")
    p = sub.add_parser("prefer", help="set the path preference for tracks and mark them pending")
    p.add_argument("path", choices=["auto", "online", "ai"])
    p.add_argument("--filter", help="only tracks whose path contains this substring")
    sub.add_parser("status", help="counts per status")
    v = sub.add_parser("serve", help="start the API + web UI")
    v.add_argument("--reload", action="store_true")
    a = ap.parse_args()

    pipeline.setup_logging()
    if a.cmd == "scan":
        from rescale_scanner import report, scan
        print(report() if a.report else scan())
    elif a.cmd == "run":
        statuses = ["pending"] + (["failed"] if a.retry_failed else []) + (["needs-review"] if a.redo_review else [])
        ids = pipeline.select_ids(statuses, a.filter, a.limit)
        print(f"{len(ids)} tracks to process")
        for n, i in enumerate(ids, 1):
            t = pipeline.process(i, embed=a.embed)
            print(f"[{n}/{len(ids)}] {t.status:15} {t.confidence!s:6} {t.path_used:6} {t.folder}/{t.filename}")
        print(pipeline.counts())
    elif a.cmd == "prefer":
        print(f"{pipeline.set_preference(a.path, a.filter)} tracks set to {a.path} and marked pending")
    elif a.cmd == "status":
        print(pipeline.counts())
    elif a.cmd == "serve":
        import uvicorn
        from rescale_core import config
        cfg = config()["api"]
        uvicorn.run("rescale_api.app:app", host=cfg["host"], port=cfg["port"], reload=a.reload)
