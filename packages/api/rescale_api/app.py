"""FastAPI: JSON API + static frontend, with a single background worker thread for processing."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rescale_core import PATHS, REPO_ROOT, STATUSES, Track, session
from rescale_scanner import scan

from rescale_api import pipeline

app = FastAPI(title="Rescale")
# ponytail: one worker thread, one GPU, in-memory queue. Lost on restart; pending rows are re-queued by /api/run.
_queue: queue.Queue[int] = queue.Queue()
_current: dict = {"id": None}
_worker_error: dict = {"msg": None}


def _worker() -> None:
    while True:
        i = _queue.get()
        _current["id"] = i
        try:
            pipeline.process(i)
        except Exception:  # process() only re-raises for a dead GPU; every queued track would fail too
            pipeline.log.exception("worker stopped; restart the server to process the rest")
            _worker_error["msg"] = "GPU error - restart the server, then re-run the failed tracks"
            while not _queue.empty():
                _queue.get_nowait()
                _queue.task_done()
            return
        finally:
            _current["id"] = None
            _queue.task_done()


@app.on_event("startup")
def _start() -> None:
    pipeline.setup_logging()
    threading.Thread(target=_worker, daemon=True, name="rescale-worker").start()


@app.get("/api/stats")
def stats() -> dict:
    current = None
    cid = _current["id"]
    if cid is not None:
        with session() as db:
            t = db.get(Track, cid)
            if t:
                current = {"id": t.id, "filename": t.filename, "folder": t.folder}
    return {"counts": pipeline.counts(), "queued": _queue.qsize(), "processing": current, "error": _worker_error["msg"]}


@app.get("/api/tracks")
def tracks(status: str | None = None, q: str | None = None) -> list[dict]:
    with session() as db:
        qry = db.query(Track).order_by(Track.folder, Track.filename)
        if status:
            qry = qry.filter(Track.status == status)
        if q:
            qry = qry.filter(Track.path.contains(q))
        out = []
        for t in qry:
            d = t.as_dict()
            d.pop("lyrics")
            out.append(d)
        return out


def _get(i: int) -> Track:
    with session() as db:
        t = db.get(Track, i)
        if not t:
            raise HTTPException(404)
        return t


@app.get("/api/tracks/{i}")
def track(i: int) -> dict:
    return _get(i).as_dict()


@app.get("/api/tracks/{i}/audio")
def audio(i: int) -> FileResponse:
    return FileResponse(_get(i).path)  # read-only streaming of the library file


@app.post("/api/tracks/{i}/reprocess")
def reprocess(i: int, path: str = Query("auto", pattern="^(auto|online|ai)$")) -> dict:
    with session() as db:
        t = db.get(Track, i)
        if not t:
            raise HTTPException(404)
        t.path_pref, t.status, t.error = path, "pending", None
        db.commit()
    _queue.put(i)
    return {"queued": i, "path": path}


@app.post("/api/run")
def run(status: str = Query("pending", pattern="^(pending|failed|needs-review)$")) -> dict:
    ids = pipeline.select_ids([status])
    for i in ids:
        _queue.put(i)
    return {"queued": len(ids)}


@app.post("/api/scan")
def rescan() -> dict:
    return scan()


app.mount("/", StaticFiles(directory=REPO_ROOT / "frontend", html=True), name="frontend")
