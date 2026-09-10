"""FastAPI: JSON API + static frontend, with a single background worker thread for processing."""
from __future__ import annotations

import mimetypes
import queue
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rescale_core import (REPO_ROOT, Library, Track, config, get_library, in_library,
                          libraries, new_id, save_libraries, session)
from rescale_scanner import remote, scan

from rescale_api import pipeline

app = FastAPI(title="Rescale")
# ponytail: one worker thread, one GPU, in-memory queue. Lost on restart; pending rows are re-queued by /api/run.
_queue: queue.Queue[tuple[int, bool | None]] = queue.Queue()
_current: dict = {"id": None}
_worker_error: dict = {"msg": None}

RUN_STATUSES = {"pending": ["pending"], "failed": ["failed"], "needs-review": ["needs-review"],
                "all": ["pending", "failed", "needs-review"]}


def _worker() -> None:
    while True:
        i, embed = _queue.get()
        _current["id"] = i
        try:
            pipeline.process(i, embed=embed)
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


def _poller() -> None:
    """Re-walk remote libraries for new files. Found files land as `pending` and stay there:
    nothing reaches the GPU without someone pressing run."""
    while True:
        time.sleep(max(1, config()["library"]["poll_minutes"]) * 60)
        for lib in libraries():
            if not (lib.is_remote and lib.auto_scan):
                continue
            try:
                found = scan(lib)
                if found["new"] or found["changed"]:
                    pipeline.log.info("poll %s: %s new, %s changed", lib.name, found["new"], found["changed"])
            except Exception as e:  # a NAS that is asleep is not an error worth killing the loop over
                pipeline.log.warning("poll %s failed: %s: %s", lib.name, type(e).__name__, e)


@app.on_event("startup")
def _start() -> None:
    pipeline.setup_logging()
    threading.Thread(target=_worker, daemon=True, name="rescale-worker").start()
    threading.Thread(target=_poller, daemon=True, name="rescale-poller").start()


def _lib(lib_id: str | None) -> Library | None:
    """Resolve ?lib=<id>. Absent or unknown = every library."""
    return get_library(lib_id)


# ---- libraries ----------------------------------------------------------------------------------


@app.get("/api/libraries")
def list_libraries() -> list[dict]:
    return [l.public() for l in libraries()]


@app.post("/api/libraries")
def save_library(url: str = Body(...), name: str = Body(""), username: str = Body(""),
                 password: str = Body(""), auto_scan: bool = Body(True), id: str = Body("")) -> dict:
    """Add or edit a library. A local path must exist; an SFTP one must connect and be a directory
    before it is saved, so a typo fails here instead of silently scanning nothing.
    Credentials are stored in data/config/libraries.json, owner-readable only."""
    url = url.strip().rstrip("/") or "/"
    libs = libraries()
    existing = next((l for l in libs if l.id == id), None)
    lib = Library(id=existing.id if existing else new_id(), name=name.strip() or Path(url).name or url,
                  url=url, username=username, password=password or (existing.password if existing else ""),
                  auto_scan=auto_scan)
    try:
        if lib.is_remote:
            remote.check(lib)
        elif not Path(lib.root).is_dir():
            raise ValueError(f"not a directory: {lib.root}")
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    if any(l.prefix == lib.prefix and l.id != lib.id for l in libs):
        raise HTTPException(400, "another library already covers that folder")
    libs = [lib if l.id == lib.id else l for l in libs] + ([] if existing else [lib])
    save_libraries(libs)
    return lib.public()


@app.delete("/api/libraries/{lib_id}")
def delete_library(lib_id: str) -> dict:
    """Forget a library and the tracks scanned from it. Nothing in the music folder is touched."""
    lib = get_library(lib_id)
    if not lib:
        raise HTTPException(404)
    with session() as db:
        dropped = in_library(db.query(Track), lib).delete(synchronize_session=False)
        db.commit()
    save_libraries([l for l in libraries() if l.id != lib_id])
    return {"deleted": lib_id, "tracks": dropped}


# ---- tracks -------------------------------------------------------------------------------------


@app.get("/api/stats")
def stats(lib: str | None = None) -> dict:
    current = None
    cid = _current["id"]
    if cid is not None:
        with session() as db:
            t = db.get(Track, cid)
            if t:
                current = {"id": t.id, "filename": t.filename, "folder": t.folder}
    scanning = next(iter(remote.progress.values()), None)
    return {"counts": pipeline.counts(_lib(lib)), "queued": _queue.qsize(), "processing": current,
            "error": _worker_error["msg"], "libraries": [l.public() for l in libraries()],
            "embed_default": config()["writer"]["embed"], "scan": scanning}


@app.get("/api/tracks")
def tracks(status: str | None = None, q: str | None = None, lib: str | None = None) -> list[dict]:
    with session() as db:
        qry = in_library(db.query(Track), _lib(lib)).order_by(Track.folder, Track.filename)
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
def audio(i: int):
    t = _get(i)
    if not remote.is_remote(t.path):
        return FileResponse(t.path)  # read-only streaming of the library file
    # ponytail: no Range support over SFTP, so the browser buffers the whole file before it can seek.
    def chunks():
        with remote.client(remote.owner(t.path)).open(str(remote.path_of(t.path)), "rb") as f:
            f.prefetch()
            while data := f.read(1 << 16):
                yield data
    return StreamingResponse(chunks(), media_type=mimetypes.guess_type(t.filename)[0] or "audio/mpeg")


@app.post("/api/tracks/{i}/reprocess")
def reprocess(i: int, path: str = Query("auto", pattern="^(auto|online|ai)$"), embed: bool | None = None) -> dict:
    with session() as db:
        t = db.get(Track, i)
        if not t:
            raise HTTPException(404)
        t.path_pref, t.status, t.error = path, "pending", None
        db.commit()
    _queue.put((i, embed))
    return {"queued": i, "path": path, "embed": embed}


@app.post("/api/run")
def run(status: str = Query("pending", pattern="^(pending|failed|needs-review|all)$"),
        lib: str | None = None, embed: bool | None = None) -> dict:
    """Queue everything in one status (or `all`: pending + failed + needs-review) for one library."""
    ids = pipeline.select_ids(RUN_STATUSES[status], _lib(lib))
    for i in ids:
        _queue.put((i, embed))
    return {"queued": len(ids)}


@app.post("/api/scan")
def rescan(lib: str | None = None) -> dict:
    """Walk one library (local folder or SFTP) recursively and upsert what it finds."""
    target = _lib(lib)
    try:
        if target:
            return scan(target) | {"library": target.name}
        results = [scan(l) for l in libraries()]  # no library selected: every one of them
        return {k: sum(r[k] for r in results) for k in ("files", "new", "changed", "unchanged", "missing")}
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")


app.mount("/", StaticFiles(directory=REPO_ROOT / "frontend", html=True), name="frontend")
