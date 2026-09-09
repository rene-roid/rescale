"""Atomic `<basename>.lrc` sidecar writer. The only code in the project that touches the library tree."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def lrc_path(audio: Path) -> Path:
    return audio.with_suffix(".lrc")


def write_lrc(audio: Path, lrc: str) -> Path:
    """Write next to the audio file: temp file in the same folder, fsync, then os.replace (atomic on POSIX)."""
    target = lrc_path(audio)
    fd, tmp = tempfile.mkstemp(prefix=f".{audio.stem}.", suffix=".lrc.tmp", dir=audio.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(lrc)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        os.unlink(tmp)
        raise
    return target
