"""SFTP library: recursive remote walk, tag probing over the wire, and per-file pull/push.

Every function takes the Library it acts on, so several remotes stay connected at once and a track
resolves back to its own server by path prefix. Everything downstream (matcher, transcriber, writer)
keeps working on local files: a remote track is pulled into data/remote/ and the result pushed back.
"""
from __future__ import annotations

import hashlib
import os
import stat
import threading
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import mutagen
import paramiko
from rescale_core import SCHEME, Library, Track, config, data_dir, library_for, session

__all__ = ["SCHEME", "check", "client", "drop", "is_remote", "owner", "path_of", "probe",
           "progress", "pull", "push", "scan_remote", "walk"]


def is_remote(path: str) -> bool:
    return path.startswith(SCHEME)


def path_of(url: str) -> PurePosixPath:
    return PurePosixPath(unquote(urlparse(url).path))


def owner(path: str) -> Library:
    """The configured library a remote track path belongs to."""
    lib = library_for(path)
    if lib is None:
        raise RuntimeError(f"no configured library covers {path} - add it back in the UI's libraries dialog")
    return lib


_conn = threading.local()  # {library prefix: (ssh, sftp)} - paramiko clients are not thread-safe

# Progress of the scans in flight, by library id, polled by /api/stats: walking 1700 files takes minutes.
progress: dict[str, dict] = {}


def client(lib: Library) -> paramiko.SFTPClient:
    """One connection per (thread, library), reopened when it drops."""
    conns = _conn.__dict__.setdefault("by_prefix", {})
    ssh, sftp = conns.get(lib.prefix, (None, None))
    if sftp is not None and ssh.get_transport() and ssh.get_transport().is_active():
        return sftp
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    # ponytail: trusts an unknown host key on first sight (homelab NAS). Pre-populate known_hosts to pin it.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(lib.host, lib.port, lib.user, lib.password or None, look_for_keys=not lib.password,
                allow_agent=not lib.password, timeout=20)
    conns[lib.prefix] = (ssh, ssh.open_sftp())
    return conns[lib.prefix][1]


def check(lib: Library) -> Library:
    """Connect and confirm the root is a directory. Raises whatever went wrong, for the UI to show."""
    _conn.__dict__.pop("by_prefix", None)  # forget a stale connection under the same prefix (edited password)
    if not stat.S_ISDIR(client(lib).stat(str(lib.root)).st_mode):
        raise ValueError(f"not a directory on {lib.host}: {lib.root}")
    return lib


def walk(lib: Library) -> list[tuple[PurePosixPath, paramiko.SFTPAttributes]]:
    """Recursive listing of audio files under the remote root. Symlinks are left alone (no loops)."""
    cfg = config()["library"]
    exts, skip = set(cfg["extensions"]), set(cfg["skip_dirs"])
    sftp, out, stack = client(lib), [], [lib.root]
    while stack:
        d = stack.pop()
        progress[lib.id] = {"phase": "listing folders", "done": len(out), "total": 0, "current": str(d)}
        for a in sftp.listdir_attr(str(d)):
            p = d / a.filename
            if stat.S_ISDIR(a.st_mode):
                if a.filename not in skip:
                    stack.append(p)
            elif stat.S_ISREG(a.st_mode) and PurePosixPath(a.filename).suffix.lower() in exts:
                out.append((p, a))
    return sorted(out, key=lambda x: str(x[0]))


def probe(lib: Library, p: PurePosixPath) -> dict:
    """Duration + tags read straight off the server: mutagen only touches headers, so no full download."""
    info = {"duration": None, "tag_title": None, "tag_artist": None, "tag_album": None}
    try:
        with client(lib).open(str(p), "rb", bufsize=1 << 16) as f:
            m = mutagen.File(f, easy=True)
        if m is not None:
            info["duration"] = getattr(m.info, "length", None)
            for k in ("title", "artist", "album"):
                v = (m.tags or {}).get(k)
                info[f"tag_{k}"] = (v[0] if isinstance(v, list) else v) or None if v else None
    except Exception:
        pass  # unreadable tags are not a reason to skip the file; the matcher falls back to the filename
    return info


def scan_remote(lib: Library) -> dict:
    """Upsert every remote audio file into the same tracks table. Unchanged files (size+mtime) are left alone."""
    try:
        return _scan(lib, walk(lib))
    finally:
        progress.pop(lib.id, None)


def _scan(lib: Library, files: list[tuple[PurePosixPath, paramiko.SFTPAttributes]]) -> dict:
    stats = {"files": len(files), "new": 0, "changed": 0, "unchanged": 0, "missing": 0}
    with session() as db:
        known = {t.path: t for t in db.query(Track).filter(Track.path.startswith(lib.prefix, autoescape=True)).all()}
        for n, (p, a) in enumerate(files, 1):
            progress[lib.id] = {"phase": "reading tags", "done": n, "total": len(files),
                                "current": str(p.relative_to(lib.root))}
            url = lib.url_for(p)
            t = known.pop(url, None)
            if t and t.size == a.st_size and t.mtime == a.st_mtime:
                stats["unchanged"] += 1
                continue
            info = probe(lib, p) | {"size": a.st_size, "mtime": float(a.st_mtime)}
            if t is None:
                t = Track(path=url, folder=str(p.parent.relative_to(lib.root)), filename=p.name)
                db.add(t)
                stats["new"] += 1
            else:
                stats["changed"] += 1
                t.status, t.lyrics, t.error = "pending", None, None
            for k, v in info.items():
                setattr(t, k, v)
        for t in known.values():  # gone from the server
            stats["missing"] += 1
            db.delete(t)
        db.commit()
    return stats


def pull(lib: Library, url: str) -> Path:
    """Download into data/remote/, keeping size+mtime so the transcriber's stem cache stays keyed the same.
    ponytail: one file at a time, deleted by drop() after processing; no library-wide mirror."""
    p = path_of(url)
    a = client(lib).stat(str(p))
    dest = data_dir("remote") / (hashlib.sha1(url.encode()).hexdigest() + p.suffix)
    st = dest.stat() if dest.exists() else None
    if not (st and st.st_size == a.st_size and int(st.st_mtime) == int(a.st_mtime)):
        tmp = dest.with_name(dest.name + ".part")
        client(lib).get(str(p), str(tmp))
        os.utime(tmp, (a.st_atime, a.st_mtime))
        os.replace(tmp, dest)
    return dest


def drop(local: Path) -> None:
    local.unlink(missing_ok=True)


def push(lib: Library, local: Path, target: PurePosixPath) -> paramiko.SFTPAttributes:
    """Upload beside the original, then rename over it, so a dropped connection cannot truncate the real file."""
    sftp = client(lib)
    tmp = target.with_name(f".{target.name}.rescale-tmp")
    sftp.put(str(local), str(tmp), confirm=True)  # confirm=True re-stats and checks the size
    try:
        try:
            sftp.posix_rename(str(tmp), str(target))
        except (AttributeError, OSError):  # server without posix-rename@openssh.com
            try:
                sftp.remove(str(target))
            except OSError:
                pass
            sftp.rename(str(tmp), str(target))
    except BaseException:
        sftp.remove(str(tmp))
        raise
    return sftp.stat(str(target))
