"""Read-only library walk + tag extraction. Only ever opens files for reading."""
from __future__ import annotations

import os
from pathlib import Path

import mutagen
from rescale_core import Track, config, library_root, session


def _tag(tags, key: str) -> str | None:
    v = (tags or {}).get(key)
    return (v[0] if isinstance(v, list) else v) or None if v else None


def probe(path: Path) -> dict:
    st = path.stat()
    info = {"size": st.st_size, "mtime": st.st_mtime, "duration": None, "tag_title": None, "tag_artist": None, "tag_album": None}
    try:
        m = mutagen.File(path, easy=True)
    except Exception:
        m = None
    if m is not None:
        info["duration"] = getattr(m.info, "length", None)
        for k in ("title", "artist", "album"):
            info[f"tag_{k}"] = _tag(m.tags, k)
    return info


def walk() -> list[Path]:
    lib = config()["library"]
    root, exts, skip = library_root(), set(lib["extensions"]), set(lib["skip_dirs"])
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        out += [Path(dirpath) / f for f in filenames if Path(f).suffix.lower() in exts]
    return sorted(out)


def scan(dry_run: bool = False) -> dict:
    """Upsert every audio file into the DB. Unchanged files (same size+mtime) are left alone."""
    root = library_root()
    files = walk()
    stats = {"files": len(files), "new": 0, "changed": 0, "unchanged": 0, "missing": 0}
    with session() as db:
        known = {t.path: t for t in db.query(Track).all()}
        for p in files:
            key = str(p)
            t = known.pop(key, None)
            st = p.stat()
            if t and t.size == st.st_size and t.mtime == st.st_mtime:
                stats["unchanged"] += 1
                continue
            info = probe(p)
            if t is None:
                t = Track(path=key, folder=str(p.parent.relative_to(root)), filename=p.name)
                db.add(t)
                stats["new"] += 1
            else:
                stats["changed"] += 1
                t.status, t.lyrics, t.error = "pending", None, None
            for k, v in info.items():
                setattr(t, k, v)
        for t in known.values():  # file gone from disk
            stats["missing"] += 1
            db.delete(t)
        if dry_run:
            db.rollback()
        else:
            db.commit()
    return stats


def report() -> str:
    """Dry-run summary of what the scanner sees: extensions, tag coverage, folders, sample rows."""
    import collections
    files = walk()
    ext = collections.Counter(p.suffix.lower() for p in files)
    folders = collections.Counter(p.parent.name for p in files)
    no_title = 0
    samples = []
    for p in files:
        info = probe(p)
        no_title += not info["tag_title"]
        if len(samples) < 15 and (len(samples) % 3 == 0 or info["tag_title"]):
            samples.append(f"  {p.parent.name} | {p.name} | title={info['tag_title']!r} artist={info['tag_artist']!r} dur={info['duration'] and round(info['duration'])}")
    return "\n".join([f"{len(files)} audio files", f"extensions: {dict(ext)}", f"folders: {dict(folders)}",
                      f"without a title tag: {no_title}", "samples:", *samples])
