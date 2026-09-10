"""Config loading + the library registry + SQLite models, shared by every package."""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]

STATUSES = ("pending", "matched-online", "ai-transcribed", "needs-review", "failed")
PATHS = ("auto", "online", "ai")
SCHEME = "sftp://"


def _env_override(section: str, key: str, current):
    raw = os.environ.get(f"RESCALE_{section}_{key}".upper())
    if raw is None:
        return current
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, (int, float)):
        return type(current)(raw)
    if isinstance(current, list):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return raw


@cache
def config() -> dict:
    path = Path(os.environ.get("RESCALE_CONFIG", REPO_ROOT / "config" / "default.toml"))
    cfg = tomllib.loads(path.read_text())
    for section, values in cfg.items():
        for key, val in values.items():
            values[key] = _env_override(section, key, val)
    data = Path(cfg["paths"]["data_dir"])
    cfg["paths"]["data_dir"] = data if data.is_absolute() else REPO_ROOT / data
    return cfg


def data_dir(sub: str) -> Path:
    """data/<sub>/, created on demand. Every package resolves its folder through this."""
    p = config()["paths"]["data_dir"] / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- library registry ---------------------------------------------------------------------------
# Which tracks belong to a library is derived from the path prefix, never stored on the row: adding a
# library adopts everything already scanned under it, and removing one leaves no orphan column behind.


@dataclass(frozen=True)
class Library:
    """One scanned source: a local folder, or a folder on another machine over SFTP."""

    id: str
    name: str
    url: str  # "/home/yuuki/Music" or "sftp://user@host[:port]/absolute/path"
    username: str = ""
    password: str = ""
    auto_scan: bool = True  # remote only: polled for new files every [library] poll_minutes

    @property
    def is_remote(self) -> bool:
        return self.url.startswith(SCHEME)

    @property
    def _parts(self):
        return urlparse(self.url)

    @property
    def host(self) -> str:
        return self._parts.hostname or ""

    @property
    def port(self) -> int:
        return self._parts.port or 22

    @property
    def user(self) -> str:
        return self.username or unquote(self._parts.username or "")

    @property
    def root(self) -> PurePosixPath | Path:
        return PurePosixPath(unquote(self._parts.path)) if self.is_remote else Path(self.url).expanduser()

    def url_for(self, p) -> str:
        """The exact string a track under this library is keyed by in the DB."""
        if not self.is_remote:
            return str(p)
        return f"{SCHEME}{quote(self.user)}@{self.host}{quote(str(p))}"  # quoted: '#' and '?' are legal in filenames

    @property
    def prefix(self) -> str:
        """What every track path under this library starts with. The trailing slash matters:
        /Music must not swallow /MusicVideos."""
        return self.url_for(self.root).rstrip("/") + "/"

    def public(self) -> dict:
        """Everything but the password, for the API."""
        return asdict(self) | {"password": "", "has_password": bool(self.password),
                               "is_remote": self.is_remote, "root": str(self.root)}


def new_id() -> str:
    return uuid4().hex[:8]


def _libraries_file() -> Path:
    return data_dir("config") / "libraries.json"


def libraries() -> list[Library]:
    """Configured libraries, seeded from the pre-registry single root on first read."""
    f = _libraries_file()
    if f.exists():
        libs = [Library(**d) for d in json.loads(f.read_text())]
    else:
        root = _default_root()
        libs = save_libraries([Library(new_id(), root.name or str(root), str(root))])
    return _apply_env_root(libs)


def _apply_env_root(libs: list[Library]) -> list[Library]:
    """RESCALE_LIBRARY_ROOT (the Docker knob) forces the first local library, without rewriting the file."""
    root = os.environ.get("RESCALE_LIBRARY_ROOT")
    if not root:
        return libs
    p = Path(root).expanduser().resolve()
    i = next((n for n, l in enumerate(libs) if not l.is_remote), None)
    if i is None:
        return [Library(new_id(), p.name or str(p), str(p))] + libs
    libs[i] = Library(libs[i].id, libs[i].name, str(p), auto_scan=libs[i].auto_scan)
    return libs


def save_libraries(libs: list[Library]) -> list[Library]:
    f = _libraries_file()
    f.touch(mode=0o600, exist_ok=True)  # SFTP passwords live in here: owner-only from the moment it exists
    f.write_text(json.dumps([asdict(l) for l in libs], indent=2))
    return libs


def get_library(lib_id: str | None) -> Library | None:
    return next((l for l in libraries() if l.id == lib_id), None) if lib_id else None


def library_for(path: str) -> Library | None:
    """The library owning a track path. Longest prefix wins, so a library nested inside another
    claims its own tracks."""
    owners = [l for l in libraries() if path.startswith(l.prefix)]
    return max(owners, key=lambda l: len(l.prefix)) if owners else None


def _default_root() -> Path:
    """Pre-registry library root: RESCALE_LIBRARY_ROOT > the root last set from the CLI > config file.
    Only used to seed the registry the first time; libraries.json owns it afterwards."""
    saved = data_dir("config") / "library_root"
    root = (os.environ.get("RESCALE_LIBRARY_ROOT")
            or (saved.read_text().strip() if saved.exists() else None)
            or config()["library"]["root"])
    return Path(root).expanduser().resolve()


def library_root() -> Path:
    """The first local library - what the CLI scans when no library is named."""
    return next((Path(l.root) for l in libraries() if not l.is_remote), _default_root())


def set_library_root(root: str) -> Path:
    """Point the first local library at `root`, adding one if there is none.
    Raises ValueError if it is not an existing directory."""
    p = Path(root).expanduser()
    if not p.is_dir():
        raise ValueError(f"not a directory: {root}")
    p = p.resolve()
    libs = libraries()
    i = next((n for n, l in enumerate(libs) if not l.is_remote), None)
    if i is None:
        libs.append(Library(new_id(), p.name or str(p), str(p)))
    else:
        libs[i] = Library(libs[i].id, libs[i].name, str(p), auto_scan=libs[i].auto_scan)
    save_libraries(libs)
    return p


def point_model_caches_at_data_dir() -> None:
    """Make torch.hub / HuggingFace / faster-whisper download into data/models/ instead of ~/.cache."""
    os.environ.setdefault("TORCH_HOME", str(data_dir("models/torch")))
    os.environ.setdefault("HF_HOME", str(data_dir("models/hf")))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class Base(DeclarativeBase):
    pass


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String, unique=True)  # absolute, or the sftp:// url; owns the library link
    folder: Mapped[str] = mapped_column(String)  # relative to its library root
    filename: Mapped[str] = mapped_column(String)
    size: Mapped[int] = mapped_column(Integer)
    mtime: Mapped[float] = mapped_column(Float)
    duration: Mapped[float | None] = mapped_column(Float)
    tag_title: Mapped[str | None] = mapped_column(String)
    tag_artist: Mapped[str | None] = mapped_column(String)
    tag_album: Mapped[str | None] = mapped_column(String)

    # what the matcher understood from filename + tags
    parsed_title: Mapped[str | None] = mapped_column(String)
    parsed_artist: Mapped[str | None] = mapped_column(String)
    is_variant: Mapped[bool] = mapped_column(Boolean, default=False)  # nightcore / sped up / slowed ...
    is_cover: Mapped[bool] = mapped_column(Boolean, default=False)  # cover / remix -> online lyrics would be wrong

    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    path_pref: Mapped[str] = mapped_column(String, default="auto")  # auto | online | ai (user choice)
    path_used: Mapped[str | None] = mapped_column(String)  # online | ai
    confidence: Mapped[float | None] = mapped_column(Float)
    match_info: Mapped[str | None] = mapped_column(Text)  # JSON: lrclib id / matched name / ratio / language ...
    lyrics: Mapped[str | None] = mapped_column(Text)  # final LRC content
    error: Mapped[str | None] = mapped_column(Text)
    lrc_written_at: Mapped[datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict:
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        for k in ("lrc_written_at", "updated_at"):
            if d[k]:
                d[k] = d[k].isoformat()
        return d


def in_library(query, lib: Library | None):
    """Scope a Track query to one library, by path prefix. None = every library."""
    return query.filter(Track.path.startswith(lib.prefix, autoescape=True)) if lib else query


@cache
def _session_factory() -> sessionmaker:
    db_path = data_dir("db") / "rescale.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def session() -> Session:
    return _session_factory()()
