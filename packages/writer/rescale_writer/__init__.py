"""`<basename>.lrc` sidecar writer, or embedded-tag writer for Navidrome-style setups.
The only code in the project that touches the library tree."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import mutagen
from mutagen.id3 import ID3, ID3NoHeaderError, USLT
from mutagen.mp4 import MP4

# One lyrics tag per container family, all of which Navidrome reads.
ID3_EXTENSIONS = {".mp3", ".wav", ".aiff", ".aif"}          # USLT frame
VORBIS_EXTENSIONS = {".flac", ".ogg", ".oga", ".opus"}      # LYRICS comment
MP4_EXTENSIONS = {".m4a", ".m4b", ".mp4"}                   # ©lyr atom
EMBEDDABLE = ID3_EXTENSIONS | VORBIS_EXTENSIONS | MP4_EXTENSIONS


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
    return audio.suffix.lower() in EMBEDDABLE


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


def _embed_id3(audio: Path, lrc: str) -> None:
    """USLT frame, plus a check that only the tag changed: an ID3 rewrite moves the audio stream, and
    a mangled one is a corrupted music file, not a failed lyric write."""
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


def _embed_tag(audio: Path, key: str, lrc: str) -> None:
    """Vorbis comment / MP4 atom. mutagen rewrites the container itself, so there is no stream to verify."""
    f = MP4(audio) if audio.suffix.lower() in MP4_EXTENSIONS else mutagen.File(audio)
    if f is None:
        raise RuntimeError(f"mutagen does not recognise {audio}")
    if f.tags is None:
        f.add_tags()
    f[key] = lrc
    f.save()


def embed_lyrics(audio: Path, lrc: str) -> Path:
    """Write `lrc` into the audio file's own lyrics tag instead of a sidecar (what Navidrome reads).
    Backs the file up first and restores it if anything goes wrong."""
    ext = audio.suffix.lower()
    if ext not in EMBEDDABLE:
        raise ValueError(f"no lyrics tag known for {ext} files")
    backup = audio.with_name(f".{audio.name}.rescale-bak")
    shutil.copy2(audio, backup)
    try:
        if ext in ID3_EXTENSIONS:
            _embed_id3(audio, lrc)
        else:
            _embed_tag(audio, "\xa9lyr" if ext in MP4_EXTENSIONS else "lyrics", lrc)
    except BaseException:
        shutil.copy2(backup, audio)
        raise
    finally:
        backup.unlink(missing_ok=True)
    return audio
