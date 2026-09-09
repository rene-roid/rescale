import os

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
    assert can_embed(pytest.importorskip("pathlib").Path("x.mp3"))
    assert not can_embed(pytest.importorskip("pathlib").Path("x.flac"))


def test_embed_restores_backup_on_failure(tmp_path, monkeypatch):
    p = _fake_mp3(tmp_path, tagged=False)
    before = p.read_bytes()

    import rescale_writer
    monkeypatch.setattr(rescale_writer, "_id3_span", lambda data: (0, 0))  # force a false mismatch: compare whole files

    with pytest.raises(RuntimeError):
        embed_lyrics(p, LRC)

    assert p.read_bytes() == before
    assert not (p.parent / f".{p.name}.rescale-bak").exists()
