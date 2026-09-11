"""Library-root override (set from the web UI) + the recursive walk it feeds."""
import shutil
import subprocess

import pytest
import rescale_core as core
from rescale_core import Track, config, library_root, session, set_library_root
from rescale_scanner import scan, walk
from rescale_writer import embed_lyrics, write_lrc


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point data_dir/ at a tmp dir so the override file never touches the real one."""
    monkeypatch.setenv("RESCALE_PATHS_DATA_DIR", str(tmp_path / "data"))
    config.cache_clear()
    core._session_factory.cache_clear()
    yield tmp_path
    config.cache_clear()
    core._session_factory.cache_clear()


def test_override_wins_over_config_and_walk_is_recursive(isolated):
    music = isolated / "music"
    (music / "artist" / "album").mkdir(parents=True)
    (music / "top.mp3").touch()
    (music / "artist" / "album" / "deep.flac").touch()
    (music / "artist" / "cover.jpg").touch()
    (music / ".mp3thumb_backup").mkdir()
    (music / ".mp3thumb_backup" / "junk.mp3").touch()

    assert library_root() != music.resolve()  # config default until the UI sets one
    assert set_library_root(str(music)) == music.resolve()
    assert library_root() == music.resolve()  # and it persists in data/config/libraries.json

    assert [p.name for p in walk()] == ["deep.flac", "top.mp3"]  # nested, no non-audio, skip_dirs honoured


def test_env_var_beats_the_stored_root(isolated, monkeypatch):
    a, b = isolated / "a", isolated / "b"
    a.mkdir(); b.mkdir()
    set_library_root(str(a))
    monkeypatch.setenv("RESCALE_LIBRARY_ROOT", str(b))
    assert library_root() == b.resolve()
    monkeypatch.delenv("RESCALE_LIBRARY_ROOT")
    assert library_root() == a.resolve()


def test_a_bad_folder_is_rejected(isolated):
    before = library_root()
    with pytest.raises(ValueError):
        set_library_root(str(isolated / "nope"))
    with pytest.raises(ValueError):
        set_library_root(str(isolated / "afile.mp3"))
    assert library_root() == before


LRC = "[ti:Faded]\n[re:Rescale]\n[00:12.50]You were the shadow to my light\n"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to make a real container")
def test_scan_adopts_tracks_that_already_have_lyrics(isolated):
    """A file that already carries synced lyrics is not pending work - whether Rescale put them there
    on a previous run or the file arrived with them."""
    music = isolated / "music"
    music.mkdir()
    made = {}
    for name in ("embedded.mp3", "sidecar.mp3", "bare.mp3", "unsynced.mp3"):
        p = music / name
        subprocess.run(["ffmpeg", "-v", "quiet", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
                        "-y", str(p)], check=True)
        made[name] = p
    embed_lyrics(made["embedded.mp3"], LRC)
    write_lrc(made["sidecar.mp3"], LRC.replace("[re:Rescale]\n", ""))  # came with the file, not ours
    embed_lyrics(made["unsynced.mp3"], "just the words, no timestamps")
    set_library_root(str(music))

    assert scan()["with_lyrics"] == 2

    with session() as db:
        by_name = {t.filename: t for t in db.query(Track)}
    assert by_name["embedded.mp3"].status == "has-lyrics"
    assert by_name["sidecar.mp3"].status == "has-lyrics"
    assert by_name["bare.mp3"].status == "pending"
    assert by_name["unsynced.mp3"].status == "pending"  # plain lyrics are not lyrics we can sync to

    assert by_name["embedded.mp3"].lyrics == LRC  # kept, so the UI can play against them
    assert '"by_rescale": true' in by_name["embedded.mp3"].match_info
    assert '"by_rescale": false' in by_name["sidecar.mp3"].match_info


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to make a real container")
def test_a_rescan_does_not_requeue_adopted_tracks(isolated):
    """The point of the whole thing: scanning twice must not put finished work back in the queue."""
    music = isolated / "music"
    music.mkdir()
    p = music / "song.mp3"
    subprocess.run(["ffmpeg", "-v", "quiet", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
                    "-y", str(p)], check=True)
    embed_lyrics(p, LRC)
    set_library_root(str(music))
    scan()

    assert scan() | {} == {"files": 1, "new": 0, "changed": 0, "unchanged": 1, "missing": 0, "with_lyrics": 0}
    with session() as db:
        assert db.query(Track).one().status == "has-lyrics"
