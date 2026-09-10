import os
import shutil
import subprocess

import mutagen
import pytest
from mutagen.id3 import ID3
from rescale_writer import can_embed, embed_lyrics, lrc_path

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
