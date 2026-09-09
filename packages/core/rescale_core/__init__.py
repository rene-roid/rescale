"""Config loading + SQLite models shared by every package."""
from __future__ import annotations

import os
import tomllib
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]

STATUSES = ("pending", "matched-online", "ai-transcribed", "needs-review", "failed")
PATHS = ("auto", "online", "ai")


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


def library_root() -> Path:
    return Path(config()["library"]["root"]).resolve()


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
    path: Mapped[str] = mapped_column(String, unique=True)  # absolute
    folder: Mapped[str] = mapped_column(String)  # relative to library root
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


@cache
def _session_factory() -> sessionmaker:
    db_path = data_dir("db") / "rescale.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def session() -> Session:
    return _session_factory()()
