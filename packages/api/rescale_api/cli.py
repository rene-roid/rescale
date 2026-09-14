"""`uv run rescale <command>`."""
from __future__ import annotations

import argparse
import json

from rescale_api import pipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="rescale", description="Synced .lrc lyrics for your music library.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="read-only library walk, upsert tracks into the DB")
    s.add_argument("--report", action="store_true", help="only print what the scanner sees, write nothing")
    s.add_argument("--root", help="folder to scan recursively; remembered for later runs and by the web UI")
    r = sub.add_parser("run", help="process tracks (pending by default) and write .lrc sidecars")
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--redo-review", action="store_true", help="also re-run tracks in needs-review")
    r.add_argument("--filter", help="only tracks whose path contains this substring")
    r.add_argument("--limit", type=int)
    r.add_argument("--embed", action="store_true", default=None,
                    help="embed lyrics into the audio file's own lyrics tag rather than a .lrc sidecar (default: config [writer] embed)")
    r.add_argument("--no-embed", dest="embed", action="store_false", help="always write a .lrc sidecar")
    g = sub.add_parser("tag", help="AI audio tags (genre / mood / tempo) for tracks that have none")
    g.add_argument("--filter", help="only tracks whose path contains this substring")
    g.add_argument("--limit", type=int)
    g.add_argument("--retag", action="store_true", help="also re-tag tracks that already have tags")
    p = sub.add_parser("prefer", help="set the path preference for tracks and mark them pending")
    p.add_argument("path", choices=["auto", "online", "ai"])
    p.add_argument("--filter", help="only tracks whose path contains this substring")
    sub.add_parser("status", help="counts per status")
    sub.add_parser("libraries", help="list the configured libraries (local folders + SFTP)")
    v = sub.add_parser("serve", help="start the API + web UI")
    v.add_argument("--reload", action="store_true")
    v.add_argument("--root", help="library folder to serve; remembered for later runs and by the web UI")
    a = ap.parse_args()

    if getattr(a, "root", None):
        from rescale_core import set_library_root
        try:
            print(f"library root: {set_library_root(a.root)}")
        except ValueError as e:
            ap.error(str(e))
    pipeline.setup_logging()
    if a.cmd == "scan":
        from rescale_scanner import report, scan
        print(report() if a.report else scan())
    elif a.cmd == "run":
        statuses = ["pending"] + (["failed"] if a.retry_failed else []) + (["needs-review"] if a.redo_review else [])
        ids = pipeline.select_ids(statuses, contains=a.filter, limit=a.limit)
        print(f"{len(ids)} tracks to process")
        for n, i in enumerate(ids, 1):
            t = pipeline.process(i, embed=a.embed)
            print(f"[{n}/{len(ids)}] {t.status:15} {t.confidence!s:6} {t.path_used:6} {t.folder}/{t.filename}")
        print(pipeline.counts())
    elif a.cmd == "tag":
        ids = pipeline.select_untagged(contains=a.filter, limit=a.limit, retag=a.retag)
        print(f"{len(ids)} tracks to tag")
        for n, i in enumerate(ids, 1):
            r = pipeline.tag_track(i)
            labels = ", ".join(f"{lbl} {sc}" for lbl, sc in json.loads(r.genres or "[]"))
            print(f"[{n}/{len(ids)}] " + (f"FAILED {r.error}" if r.error else
                                          f"{r.bpm:>5} bpm  {r.variant or '-':9} {labels}"))
    elif a.cmd == "prefer":
        print(f"{pipeline.set_preference(a.path, a.filter)} tracks set to {a.path} and marked pending")
    elif a.cmd == "status":
        print(pipeline.counts())
    elif a.cmd == "libraries":
        from rescale_core import libraries
        for l in libraries():
            print(f"{l.id}  {l.name:20} {l.url}{'  (auto-scan)' if l.is_remote and l.auto_scan else ''}")
    elif a.cmd == "serve":
        import uvicorn
        from rescale_core import config
        cfg = config()["api"]
        uvicorn.run("rescale_api.app:app", host=cfg["host"], port=cfg["port"], reload=a.reload)
