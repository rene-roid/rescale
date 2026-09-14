"""`<basename>.lrc` sidecar writer, or embedded-tag writer for Navidrome-style setups.
The only code in the project that touches the library tree."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mutagen
from mutagen.id3 import ID3, ID3NoHeaderError, TCON, USLT
from mutagen.mp4 import MP4

# One lyrics tag per container family, all of which Navidrome reads.
ID3_EXTENSIONS = {".mp3", ".wav", ".aiff", ".aif"}          # USLT frame
VORBIS_EXTENSIONS = {".flac", ".ogg", ".oga", ".opus"}      # LYRICS comment
MP4_EXTENSIONS = {".m4a", ".m4b", ".mp4"}                   # ©lyr atom
EMBEDDABLE = ID3_EXTENSIONS | VORBIS_EXTENSIONS | MP4_EXTENSIONS


LYRIC_KEY = {"id3": "USLT::und", "vorbis": "lyrics", "mp4": "\xa9lyr"}
SYNCED = re.compile(r"^\[\d+:\d{2}", re.M)  # a real LRC line; a plain lyric dump has none
MARK = "[re:Rescale]"  # build() stamps this, so lyrics we wrote ourselves are recognisable later


def lrc_path(audio: Path) -> Path:
    return audio.with_suffix(".lrc")


def _family(suffix: str) -> str | None:
    suffix = suffix.lower()
    return ("id3" if suffix in ID3_EXTENSIONS else "vorbis" if suffix in VORBIS_EXTENSIONS
            else "mp4" if suffix in MP4_EXTENSIONS else None)


def is_synced(lrc: str | None) -> bool:
    """Timestamped lyrics. Unsynced lyrics do not count - a plain lyric dump is what Rescale replaces."""
    return bool(lrc and SYNCED.search(lrc))


def read_embedded(target, suffix: str) -> str | None:
    """Lyrics already in the file's own tag. `target` is a path, or an open file object so a remote
    track can be read over SFTP without downloading it."""
    family = _family(suffix)
    if family is None:
        return None
    try:
        f = MP4(target) if family == "mp4" else mutagen.File(target)
        if f is None or f.tags is None:
            return None
        v = f.tags.get(LYRIC_KEY[family]) if family != "vorbis" else f.get("lyrics")
        if family == "id3" and v is None:  # any USLT frame, whatever its language/description
            v = next(iter(f.tags.getall("USLT")), None)
    except Exception:
        return None  # an unreadable tag just means "no lyrics found", never a scan failure
    if v is None:
        return None
    text = getattr(v, "text", v)
    return text[0] if isinstance(text, list) else text


def existing_lyrics(audio: Path) -> tuple[str, str] | None:
    """Synced lyrics the track already carries: (lrc, "embedded" | "sidecar"), or None."""
    embedded = read_embedded(audio, audio.suffix)
    if is_synced(embedded):
        return embedded, "embedded"
    side = lrc_path(audio)
    if side.is_file():
        text = side.read_text(encoding="utf-8", errors="replace")
        if is_synced(text):
            return text, "sidecar"
    return None


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


def _verify_stream(audio: Path, before: bytes) -> None:
    """Check that only the tag changed: an ID3 rewrite moves the audio stream, and a mangled one is a
    corrupted music file, not a failed tag write."""
    after = audio.read_bytes()
    bh, bt = _id3_span(before)
    ah, at = _id3_span(after)
    if before[bh:len(before) - bt] != after[ah:len(after) - at]:
        raise RuntimeError(f"audio stream changed while writing tags into {audio}")


def _embed_id3(audio: Path, lrc: str) -> None:
    """USLT frame, plus the stream check."""
    before = audio.read_bytes()
    try:
        tags = ID3(audio)
    except ID3NoHeaderError:
        tags = ID3()
    tags.delall("USLT")
    tags.add(USLT(encoding=3, lang="und", desc="", text=lrc))
    tags.save(audio, v2_version=3)
    _verify_stream(audio, before)


def _embed_tag(audio: Path, key: str, lrc: str) -> None:
    """Vorbis comment / MP4 atom. mutagen rewrites the container itself, so there is no stream to verify."""
    f = MP4(audio) if audio.suffix.lower() in MP4_EXTENSIONS else mutagen.File(audio)
    if f is None:
        raise RuntimeError(f"mutagen does not recognise {audio}")
    if f.tags is None:
        f.add_tags()
    f[key] = lrc
    f.save()


@contextmanager
def _guarded(audio: Path):
    """Back the file up for the length of a tag write and put it back if anything goes wrong.
    A failed tag write must never leave a damaged music file behind."""
    backup = audio.with_name(f".{audio.name}.rescale-bak")
    shutil.copy2(audio, backup)
    try:
        yield
    except BaseException:
        shutil.copy2(backup, audio)
        raise
    finally:
        backup.unlink(missing_ok=True)


def embed_lyrics(audio: Path, lrc: str) -> Path:
    """Write `lrc` into the audio file's own lyrics tag instead of a sidecar (what Navidrome reads)."""
    ext = audio.suffix.lower()
    if ext not in EMBEDDABLE:
        raise ValueError(f"no lyrics tag known for {ext} files")
    with _guarded(audio):
        if ext in ID3_EXTENSIONS:
            _embed_id3(audio, lrc)
        else:
            _embed_tag(audio, "\xa9lyr" if ext in MP4_EXTENSIONS else "lyrics", lrc)
    return audio


def embed_genres(audio: Path, genres: list[str]) -> Path:
    """Write the genre tag - the field Navidrome groups by. Every container has a first-class genre
    field, so mutagen's easy interface covers them all except wav and aiff, whose tags are raw ID3
    frames living in a RIFF chunk. Those go through mutagen's container class, never a bare ID3 save:
    an ID3 tag written straight to a wav lands in front of the RIFF header and nothing can read it."""
    ext = audio.suffix.lower()
    if ext not in EMBEDDABLE:
        raise ValueError(f"no genre tag known for {ext} files")
    if not genres:
        return audio
    with _guarded(audio):
        riff = ext in ID3_EXTENSIONS and ext != ".mp3"
        before = audio.read_bytes() if ext == ".mp3" else b""
        f = mutagen.File(audio) if riff else mutagen.File(audio, easy=True)
        if f is None:
            raise RuntimeError(f"mutagen does not recognise {audio}")
        if f.tags is None:
            f.add_tags()
        if riff:
            f.tags.setall("TCON", [TCON(encoding=3, text=genres)])
        else:
            f["genre"] = genres
        f.save()
        if before:
            _verify_stream(audio, before)
    return audio
