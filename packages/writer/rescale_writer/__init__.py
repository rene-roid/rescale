"""`<basename>.lrc` sidecar writer, or embedded-tag writer for Navidrome-style setups.
The only code in the project that touches the library tree."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from mutagen.id3 import ID3, ID3NoHeaderError, USLT

# ID3 is the only tag format this writes into (per the USLT frame the feature was asked for).
# Other library formats (flac/m4a/ogg/opus) fall back to the .lrc sidecar.
ID3_EXTENSIONS = {".mp3", ".wav", ".aiff", ".aif"}


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


def can_embed(audio: Path) -> bool:
    return audio.suffix.lower() in ID3_EXTENSIONS


def _id3_span(data: bytes) -> tuple[int, int]:
    """(header bytes, trailer bytes) to strip off `data` to get at the raw audio stream.
    ponytail: doesn't account for a trailing APEv2 tag; not present in this library."""
    head = 0
    if data[:3] == b"ID3":
        size = 0
        for b in data[6:10]:
            size = (size << 7) | (b & 0x7F)  # synchsafe integer
        head = 10 + size
    tail = 128 if data[-128:-125] == b"TAG" else 0
    return head, tail


def embed_lyrics(audio: Path, lrc: str) -> Path:
    """Write `lrc` into an ID3 USLT frame instead of a sidecar file (what Navidrome reads).
    Backs the file up first, and verifies the audio stream is byte-for-byte untouched -
    only the tag changed - before dropping the backup; restores it otherwise."""
    backup = audio.with_name(f".{audio.name}.rescale-bak")
    shutil.copy2(audio, backup)
    try:
        before = audio.read_bytes()
        try:
            tags = ID3(audio)
        except ID3NoHeaderError:
            tags = ID3()
        tags.delall("USLT")
        tags.add(USLT(encoding=3, lang="und", desc="", text=lrc))
        tags.save(audio, v2_version=3)

        after = audio.read_bytes()
        bh, bt = _id3_span(before)
        ah, at = _id3_span(after)
        if before[bh:len(before) - bt] != after[ah:len(after) - at]:
            raise RuntimeError(f"audio stream changed while embedding lyrics into {audio}")
    except BaseException:
        shutil.copy2(backup, audio)
        raise
    finally:
        backup.unlink(missing_ok=True)
    return audio
