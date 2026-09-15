import os
import shutil
import subprocess

import mutagen
import pytest
from mutagen.id3 import ID3
from rescale_writer import (can_embed, embed_lyrics, embed_tags, existing_lyrics, is_synced,
                            lrc_path, write_lrc)

LRC = "[ti:Faded]\n[00:12.50]You were the shadow to my light\n"


def _fake_mp3(tmp_path, tagged: bool) -> "os.PathLike":
    p = tmp_path / "song.mp3"
    audio = os.urandom(4096)  # stand-in for real MPEG frames: ID3 never looks past its own header
    if tagged:
        ID3().save(p, v2_version=3)  # id3-only file first, so `audio` below is untouched payload
        p.write_bytes(p.read_bytes() + audio)
    else:
        p.write_bytes(audio)
    return p


@pytest.mark.parametrize("tagged", [False, True])
def test_embed_preserves_audio_and_writes_uslt(tmp_path, tagged):
    p = _fake_mp3(tmp_path, tagged)
    before = p.read_bytes()

    embed_lyrics(p, LRC)

    tags = ID3(p)
    assert tags.getall("USLT")[0].text == LRC
    assert not lrc_path(p).exists()
    assert not (p.parent / f".{p.name}.rescale-bak").exists()

    after = p.read_bytes()
    assert before[-4096:] == after[-4096:]  # the "audio" payload, byte-for-byte


def test_can_embed():
    from pathlib import Path
    assert all(can_embed(Path("x" + e)) for e in (".mp3", ".flac", ".ogg", ".opus", ".m4a"))
    assert not can_embed(Path("x.wma"))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to make a real container")
@pytest.mark.parametrize("ext, key", [(".flac", "lyrics"), (".ogg", "lyrics"), (".opus", "lyrics"),
                                      (".m4a", "\xa9lyr")])
def test_embed_round_trips_every_non_id3_format(tmp_path, ext, key):
    """The formats that used to fall back to a sidecar. A real container, so this fails if mutagen
    cannot actually carry the tag rather than only if the dispatch picks the wrong key."""
    p = tmp_path / ("song" + ext)
    subprocess.run(["ffmpeg", "-v", "quiet", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
                    "-y", str(p)], check=True)

    assert embed_lyrics(p, LRC) == p

    f = mutagen.File(p)
    assert f[key][0] == LRC
    assert f.info.length > 0  # still a playable file, not a tag with debris attached
    assert not lrc_path(p).exists()
    assert not (p.parent / f".{p.name}.rescale-bak").exists()


def test_embed_restores_backup_on_failure(tmp_path, monkeypatch):
    p = _fake_mp3(tmp_path, tagged=False)
    before = p.read_bytes()

    import rescale_writer
    monkeypatch.setattr(rescale_writer, "_id3_span", lambda data: (0, 0))  # force a false mismatch: compare whole files

    with pytest.raises(RuntimeError):
        embed_lyrics(p, LRC)

    assert p.read_bytes() == before
    assert not (p.parent / f".{p.name}.rescale-bak").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to make a real container")
@pytest.mark.parametrize("ext", [".mp3", ".flac", ".ogg", ".opus", ".m4a", ".wav"])
def test_lyrics_written_into_a_file_are_found_again(tmp_path, ext):
    """Every format we embed into, we must be able to read back - otherwise a rescan re-runs the
    whole library through the GPU to redo work that is already sitting in the files.

    The ffmpeg check is the point of the wav case: mutagen reads its own prepended tag back happily,
    so a round-trip alone still passes on a wav that ffmpeg can no longer open at all."""
    p = tmp_path / ("song" + ext)
    subprocess.run(["ffmpeg", "-v", "quiet", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
                    "-y", str(p)], check=True)
    assert existing_lyrics(p) is None

    embed_lyrics(p, LRC)
    assert existing_lyrics(p) == (LRC, "embedded")
    assert subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"],
                          capture_output=True).returncode == 0


def test_a_sidecar_counts_too_and_plain_lyrics_do_not(tmp_path):
    p = _fake_mp3(tmp_path, tagged=False)
    assert existing_lyrics(p) is None

    write_lrc(p, "these are just the words\nwith no timestamps at all\n")
    assert existing_lyrics(p) is None  # unsynced: exactly what Rescale exists to replace

    write_lrc(p, LRC)
    assert existing_lyrics(p) == (LRC, "sidecar")


def test_is_synced():
    assert is_synced("[00:12.50]a line")
    assert is_synced("[ti:Faded]\n[00:12.50]a line")
    assert not is_synced("[ti:Faded]\njust words")
    assert not is_synced("") and not is_synced(None)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to make a real container")
@pytest.mark.parametrize("ext", [".mp3", ".flac", ".ogg", ".opus", ".m4a", ".wav"])
def test_embed_tags_round_trips_and_leaves_the_file_playable(tmp_path, ext):
    """Genre, mood and bpm into a real container, decoded again afterwards. The wav case is the one
    that matters: an ID3 tag written straight to a wav lands ahead of the RIFF header and ffmpeg stops
    reading it - every other format already had an easy path, wav does not."""
    p = tmp_path / ("song" + ext)
    subprocess.run(["ffmpeg", "-v", "quiet", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
                    "-y", str(p)], check=True)

    assert embed_tags(p, ["nightcore", "hyperpop"], ["energetic", "uplifting"], 154.4) == p

    if ext in (".mp3", ".wav"):
        tags = mutagen.File(p).tags
        genre, mood, bpm = tags.getall("TCON")[0].text, tags.getall("TMOO")[0].text, tags.getall("TBPM")[0].text
    elif ext == ".m4a":
        f = mutagen.File(p)
        genre, bpm = f["\xa9gen"], [str(v) for v in f["tmpo"]]
        mood = [v.decode() for v in f["----:com.apple.itunes:mood"]]
    else:
        f = mutagen.File(p, easy=True)
        genre, mood, bpm = f["genre"], f["mood"], f["bpm"]
    assert list(genre) == ["nightcore", "hyperpop"]
    assert list(mood) == ["energetic", "uplifting"]
    assert list(bpm) == ["154"]  # rounded: none of these fields carry fractional BPM
    assert subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"],
                          capture_output=True).returncode == 0
    assert not (p.parent / f".{p.name}.rescale-bak").exists()


def test_embed_tags_is_a_no_op_with_nothing_to_write(tmp_path):
    p = _fake_mp3(tmp_path, tagged=False)
    before = p.read_bytes()
    assert embed_tags(p, [], [], None) == p
    assert p.read_bytes() == before
