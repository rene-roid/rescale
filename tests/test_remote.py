"""The SFTP library: url round-trip, recursive walk filtering, and upserting remote tracks."""
import shutil
import stat
from pathlib import PurePosixPath

import pytest
import rescale_core as core
from rescale_core import Library, Track, config, library_for, save_libraries, session
from rescale_scanner import remote, scan
from rescale_scanner.remote import SCHEME
from rescale_writer import embed_lyrics, read_embedded

NAS = Library("nas1", "NAS", "sftp://yuuki@nas/mnt/data/music", password="gamer")


class Attr:
    def __init__(self, filename, mode=stat.S_IFREG, size=10, mtime=1.0):
        self.filename, self.st_mode, self.st_size, self.st_mtime = filename, mode, size, mtime


class FakeSFTP:
    """Directory tree as {path: [Attr, ...]}."""
    def __init__(self, tree):
        self.tree = tree

    def listdir_attr(self, path):
        return self.tree[path]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("RESCALE_PATHS_DATA_DIR", str(tmp_path / "data"))
    config.cache_clear()
    core._session_factory.cache_clear()
    save_libraries([NAS])
    monkeypatch.setattr(remote, "probe",
                        lambda lib, p, size: ({"duration": 1.0, "tag_title": None, "tag_artist": None, "tag_album": None}, None))
    yield tmp_path
    config.cache_clear()
    core._session_factory.cache_clear()


DIR = stat.S_IFDIR
TREE = {
    "/mnt/data/music": [Attr("a.mp3"), Attr("a.lrc"), Attr("cover.jpg"), Attr("Artist", DIR), Attr(".mp3thumb_backup", DIR)],
    "/mnt/data/music/Artist": [Attr("deep #1?.flac")],
    "/mnt/data/music/.mp3thumb_backup": [Attr("junk.mp3")],
}


def test_walk_is_recursive_and_filtered(isolated, monkeypatch):
    monkeypatch.setattr(remote, "client", lambda lib: FakeSFTP(TREE))
    files, sidecars = remote.walk(NAS)
    assert [str(p) for p, _ in files] == ["/mnt/data/music/Artist/deep #1?.flac", "/mnt/data/music/a.mp3"]
    # .lrc files come out of the same listing, so "does this track already have lyrics" is free
    assert {str(p) for p in sidecars} == {"/mnt/data/music/a.lrc"}


def test_url_survives_characters_that_break_urls(isolated):
    url = NAS.url_for(PurePosixPath("/mnt/data/music/Artist/deep #1?.flac"))
    assert remote.is_remote(url)
    assert remote.path_of(url) == PurePosixPath("/mnt/data/music/Artist/deep #1?.flac")


def test_scan_remote_upserts_and_forgets_gone_files(isolated, monkeypatch):
    monkeypatch.setattr(remote, "client", lambda lib: FakeSFTP(TREE))
    assert remote.scan_remote(NAS) | {} == {"files": 2, "new": 2, "changed": 0, "unchanged": 0,
                                           "missing": 0, "with_lyrics": 0}
    assert remote.scan_remote(NAS)["unchanged"] == 2  # same size+mtime -> left alone

    smaller = dict(TREE, **{"/mnt/data/music": [Attr("a.mp3", size=99)]})
    monkeypatch.setattr(remote, "client", lambda lib: FakeSFTP(smaller))
    assert remote.scan_remote(NAS) | {} == {"files": 1, "new": 0, "changed": 1, "unchanged": 0,
                                           "missing": 1, "with_lyrics": 0}
    with session() as db:
        t = db.query(Track).one()
        assert t.path.startswith(SCHEME + "yuuki@nas/mnt/data/music") and t.filename == "a.mp3" and t.folder == "."


def test_scanning_one_library_leaves_the_others_alone(isolated, monkeypatch):
    """Membership is derived from the path prefix, so an empty local library must not delete NAS rows."""
    monkeypatch.setattr(remote, "client", lambda lib: FakeSFTP(TREE))
    remote.scan_remote(NAS)
    music = isolated / "music"
    music.mkdir()
    local = Library("loc1", "Music", str(music))
    save_libraries([NAS, local])

    assert scan(local)["missing"] == 0
    with session() as db:
        assert db.query(Track).count() == 2


def test_a_library_owns_a_track_by_longest_prefix(isolated):
    """/Music must not swallow /MusicVideos, and a library nested in another keeps its own tracks."""
    outer = Library("o", "outer", "/srv/media")
    inner = Library("i", "inner", "/srv/media/nightcore")
    other = Library("v", "videos", "/srv/mediavideos")
    save_libraries([outer, inner, other, NAS])

    assert library_for("/srv/media/pop/x.mp3").id == "o"
    assert library_for("/srv/media/nightcore/x.mp3").id == "i"      # inner wins over outer
    assert library_for("/srv/mediavideos/x.mp3").id == "v"          # prefix must not run past the folder
    assert library_for(NAS.url_for(PurePosixPath("/mnt/data/music/a.mp3"))).id == "nas1"
    assert library_for("/somewhere/else.mp3") is None


@pytest.mark.parametrize("embed, expected", [
    (True, "/mnt/data/music/Artist/song.mp3"),      # the audio file itself, tag rewritten
    (False, "/mnt/data/music/Artist/song.lrc"),     # sidecar beside the original
])
def test_write_pushes_to_the_original_name_not_the_hashed_local_copy(isolated, monkeypatch, embed, expected):
    from rescale_api import pipeline
    local = isolated / "0ea6bab.mp3"  # what pull() names it
    local.write_bytes(b"audio")
    pushed = {}
    monkeypatch.setattr(pipeline, "write_lyrics",
                        lambda audio, lrc, e: audio if e else audio.with_suffix(".lrc"))
    def fake_push(lib, f, target):
        pushed["t"] = str(target)
        pushed["lib"] = lib.id
        return Attr("x", size=7, mtime=9.0)
    monkeypatch.setattr(remote, "push", fake_push)
    t = Track(path=NAS.url_for(PurePosixPath("/mnt/data/music/Artist/song.mp3")), size=1, mtime=1.0)
    pipeline._write(t, local, "[00:00.00] hi", embed)
    assert pushed["t"] == expected
    assert pushed["lib"] == "nas1"  # resolved back to its own server by prefix
    assert (t.size, t.mtime) == ((7, 9.0) if embed else (1, 1.0))  # only an embed changes the file itself


def test_tags_parse_the_same_from_the_ends_of_a_file_as_from_all_of_it(tmp_path):
    """Tag reading pulls only the head and tail of a remote file, because letting mutagen seek through
    the audio stream costs a 5 ms round trip per few KB - 13 s on one 50 MB file. The bytes it does
    get must produce identical tags, and an identical duration: mutagen derives the length of a CBR
    mp3 from the file size, so the window has to keep reporting the real one."""
    import subprocess
    import mutagen
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg to make a real container")
    p = tmp_path / "song.mp3"
    subprocess.run(["ffmpeg", "-v", "quiet", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    "-metadata", "title=Faded", "-metadata", "artist=Alan Walker", "-y", str(p)], check=True)
    embed_lyrics(p, "[ti:Faded]\n[00:02.50]You were the shadow\n")

    whole = mutagen.File(p, easy=True)
    raw = p.read_bytes()
    size = len(raw)
    window = remote._Ends(raw[:4096], raw[-512:], size)  # a head far smaller than the audio stream
    part = mutagen.File(window, easy=True)

    assert part.tags["title"] == whole.tags["title"] == ["Faded"]
    assert part.tags["artist"] == whole.tags["artist"] == ["Alan Walker"]
    assert part.info.length == whole.info.length
    window.seek(0)
    assert read_embedded(window, ".mp3") == read_embedded(p, ".mp3")


def test_the_window_reports_the_real_size_and_eofs_inside_the_audio():
    w = remote._Ends(b"HEAD", b"TAIL", 1000)
    assert w.seek(0, 2) == 1000                      # seek-to-end must give the true length
    assert w.seek(0) == 0 and w.read(4) == b"HEAD"
    assert w.seek(996) == 996 and w.read(4) == b"TAIL"
    assert w.seek(500) == 500 and w.read(10) == b""  # the audio stream: EOF, never a network read


@pytest.mark.parametrize("header, expected", [
    ("bytes=0-",          (0, 999)),    # what a browser sends first for an <audio> element
    ("bytes=100-199",     (100, 199)),
    ("bytes=500-99999",   (500, 999)),  # past the end: clamped, not an error
    ("bytes=-128",        (872, 999)),  # a suffix range, counted back from the end
    ("bytes=0-0",         (0, 0)),
    ("bytes=0-, 500-600", (0, 999)),    # multi-range: only the first is served
    (None,                None),        # no header at all: send the whole file
    ("",                  None),
    ("seconds=1-2",       None),
    ("bytes=abc",         None),
])
def test_range_header_parsing(header, expected):
    """Remote playback hangs off this: a misparsed offset serves the wrong bytes, which the browser
    decodes as noise rather than reporting as an error."""
    from rescale_api.app import byte_range
    assert byte_range(header, 1000) == expected
