"""Library-root override (set from the web UI) + the recursive walk it feeds."""
import pytest
import rescale_core as core
from rescale_core import config, library_root, set_library_root
from rescale_scanner import walk


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point data_dir/ at a tmp dir so the override file never touches the real one."""
    monkeypatch.setenv("RESCALE_PATHS_DATA_DIR", str(tmp_path / "data"))
    config.cache_clear()
    yield tmp_path
    config.cache_clear()


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
    assert library_root() == music.resolve()  # and it persists in data/config/library_root

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
