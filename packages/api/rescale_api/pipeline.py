"""Orchestration: pick a path per track, run it, write the sidecar, record status in the DB."""
from __future__ import annotations

import collections
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from rescale_core import STATUSES, Library, Track, TrackTags, config, data_dir, in_library, session
from rescale_matcher import candidates, lyric_similarity, parse
from rescale_scanner import remote
from rescale_rescaler import build, fit, lines, rescale
from rescale_writer import can_embed, embed_lyrics, write_lrc

log = logging.getLogger("rescale")


def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(data_dir("logs") / "rescale.log")])


def decide(t: Track) -> str:
    """online | ai. Explicit per-track preference wins; covers/remixes always go AI; shifted variants use the config default."""
    if t.path_pref != "auto":
        return t.path_pref
    if t.is_cover:
        return "ai"
    if t.is_variant:
        return config()["matcher"]["variant_default_path"]
    return "online"


@contextmanager
def _localized(t: Track):
    """Yield a local file for the track. Everything downstream (whisper, demucs, the writer) needs a real
    path, so an sftp:// track is pulled down first and the copy is dropped again afterwards."""
    if not remote.is_remote(t.path):
        yield Path(t.path)
        return
    local = remote.pull(remote.owner(t.path), t.path)
    try:
        yield local
    finally:
        remote.drop(local)


def _drop_stem(audio: Path) -> None:
    """Throw away the vocal stem now the track is done with it, unless it is being kept on purpose."""
    if config()["transcriber"]["keep_stems"]:
        return
    try:
        from rescale_transcriber import drop_stem  # module import is light; torch loads on first use
        if freed := drop_stem(audio):
            log.debug("  freed %.0f MB of stem", freed / 2**20)
    except Exception:  # a stem we cannot delete is not a reason to fail a track that succeeded
        log.debug("  could not drop the stem for %s", audio, exc_info=True)


def run_online(t: Track, p, audio: Path) -> tuple[str, str, float, dict] | None:
    cfg = config()["matcher"]
    cands = candidates(p, t.duration)[:3]
    if not cands:
        log.info("no LRCLIB candidates for %s", t.filename)
        return None
    info: dict = {}
    if cfg["verify"]:
        # Listen before trusting a title match: overlap between what Whisper hears and each candidate's lyrics.
        from rescale_transcriber import observed_lines
        obs = observed_lines(audio)
        heard = " ".join(x for _, x in obs)
        for m in cands:
            m.lyric_sim = lyric_similarity(heard, m.record["plainLyrics"])
        cands.sort(key=lambda m: (-(m.lyric_sim or 0), -m.confidence))
        m = cands[0]
        log.info("  candidates: %s", [(c.record["trackName"], c.record["artistName"], c.confidence, c.lyric_sim) for c in cands])
        info["heard"] = heard[:400]
        if (m.lyric_sim or 0) < cfg["min_lyric_sim"]:
            return None  # what the audio says is not this song at all
        conf = 0.5 * m.confidence + 0.5 * min(1.0, m.lyric_sim / 0.5)
        f = fit(lines(m.record["syncedLyrics"]), obs)
        info["fit"] = f
        if f and not 0.5 <= f[0] <= 2.0:
            log.info("  timing fit is nonsense (%s): heavily cut or restructured -> AI", f)
            return None
        if f:
            ratio, offset = f[0], f[1]  # timing measured from the audio itself: speed change + any trim/lead-in
            if f[3] > 1.5:  # lines paired inconsistently (repetitive chorus, missed lines): plausible but unproven
                conf -= 0.15
        else:
            ratio, offset = (m.ratio if t.is_variant else 1.0), 0.0
            conf -= 0.1  # no measured timing; duration ratio is a guess for variants
    else:
        m = cands[0]
        if m.confidence < cfg["review"]:
            log.info("no online match for %s (best %s)", t.filename, (m.record["trackName"], m.record["artistName"], m.confidence))
            return None
        conf, ratio, offset = m.confidence, (m.ratio if t.is_variant else 1.0), 0.0
    if abs(ratio - 1) < 0.03 and abs(offset) < 0.3:
        ratio, offset = 1.0, 0.0  # encoder jitter, not a speed change
    r = m.record
    lrc = build(lines(rescale(r["syncedLyrics"], ratio, offset)), title=r["trackName"], artist=r["artistName"])
    conf = round(min(1.0, conf), 3)
    status = "matched-online" if conf >= cfg["accept"] else "needs-review"
    if (m.lyric_sim or 1.0) < cfg["review_lyric_sim"]:
        # Heard enough to rule out a different song, not enough to trust it: write it, but flag it.
        # info["heard"] is in the detail pane next to the lyrics so the call is yours, not a threshold's.
        status = "needs-review"
    info.update({"lrclib_id": r["id"], "track": r["trackName"], "artist": r["artistName"], "album": r.get("albumName"),
                 "orig_duration": r["duration"], "ratio": ratio, "offset": offset, "title_sim": m.title_sim, "lyric_sim": m.lyric_sim})
    return lrc, status, conf, info


def run_ai(t: Track, p, audio: Path) -> tuple[str, str, float, dict]:
    from rescale_transcriber import transcribe_to_lrc  # heavy import, only when needed
    lrc, info = transcribe_to_lrc(audio, p.title, p.artist or t.tag_artist)
    status = "ai-transcribed" if info["score"] >= config()["transcriber"]["review_below"] else "needs-review"
    return lrc, status, info["score"], info


def write_lyrics(audio: Path, lrc: str, embed: bool) -> Path:
    """Embed into the audio file's USLT tag (Navidrome-friendly) if asked and the format supports it,
    else fall back to a .lrc sidecar."""
    if embed and can_embed(audio):
        return embed_lyrics(audio, lrc)
    return write_lrc(audio, lrc)


def _write(t: Track, audio: Path, lrc: str, embed: bool) -> None:
    """Write the lyrics next to (or into) the track, pushing the result back to the server for remote ones."""
    written = write_lyrics(audio, lrc, embed)
    embedded = written.name == audio.name  # else it is the .lrc sidecar beside it
    if remote.is_remote(t.path):
        original = remote.path_of(t.path)  # the local copy is named by hash, so the target comes from the url
        st = remote.push(remote.owner(t.path), written,
                         original if embedded else original.with_suffix(written.suffix))
    else:
        st = written.stat()
    if embedded:  # the audio file itself changed: record its new size+mtime or the next scan re-queues it
        t.size, t.mtime = st.st_size, float(st.st_mtime)


def process(track_id: int, embed: bool | None = None) -> Track:
    if embed is None:
        embed = config()["writer"]["embed"]
    with session() as db:
        t = db.get(Track, track_id)
        p = parse(t.tag_title, t.filename, t.tag_artist, t.folder)
        t.parsed_title, t.parsed_artist, t.is_variant, t.is_cover = p.title, p.artist, p.is_variant, p.is_cover
        path = decide(t)
        log.info("[%s] %s/%s  title=%r artist=%r variant=%s cover=%s", path, t.folder, t.filename, p.title, p.artist, p.is_variant, p.is_cover)
        fatal = None
        try:
            with _localized(t) as audio:
                try:
                    result = None
                    if path == "online":
                        result = run_online(t, p, audio)
                        if result is None and t.path_pref == "auto":
                            path = "ai"
                    if path == "ai":
                        result = run_ai(t, p, audio)
                    if result is None:
                        raise LookupError("no online lyrics match (path preference is online-only)")
                    lrc, status, conf, info = result
                    _write(t, audio, lrc, embed)
                finally:
                    _drop_stem(audio)
            t.lyrics, t.status, t.confidence, t.match_info, t.error = lrc, status, conf, json.dumps(info), None
            t.lrc_written_at = datetime.now(timezone.utc)
            log.info("  -> %s conf=%s %s", status, conf, info)
        except Exception as e:  # noqa: BLE001 - one bad track must not stop the batch
            log.exception("  -> failed: %s", e)
            t.status, t.error = "failed", f"{type(e).__name__}: {e}"
            # ...but a CUDA error is not a bad track: it poisons the process, so every later track would
            # fail the same way. Stop instead of marking the rest of the library failed.
            fatal = e if "cuda" in str(e).lower() else None
        t.path_used = path
        db.commit()
        if fatal:
            raise RuntimeError(f"GPU is unusable in this process ({fatal}) - stopping. Restart and `run --retry-failed`.") from fatal
        return t


def select_ids(statuses=("pending",), lib: Library | None = None, contains: str | None = None,
               limit: int | None = None) -> list[int]:
    with session() as db:
        q = in_library(db.query(Track.id), lib).filter(Track.status.in_(statuses)).order_by(Track.folder, Track.filename)
        if contains:
            q = q.filter(Track.path.contains(contains))
        if limit:
            q = q.limit(limit)
        return [i for (i,) in q]


def set_preference(pref: str, contains: str | None = None) -> int:
    """Per-track path preference (auto/online/ai); matching tracks go back to pending."""
    with session() as db:
        q = db.query(Track)
        if contains:
            q = q.filter(Track.path.contains(contains))
        n = 0
        for t in q:
            t.path_pref, t.status = pref, "pending"
            n += 1
        db.commit()
        return n


def counts(lib: Library | None = None) -> dict:
    """Per-status totals for one library, or the whole DB when none is given."""
    with session() as db:
        rows = in_library(db.query(Track.status), lib)
        n = collections.Counter(s for (s,) in rows)
        return {s: n.get(s, 0) for s in STATUSES}


# ---- tagging -------------------------------------------------------------------------------------


def tag_track(track_id: int) -> TrackTags:
    """AI audio tags for one track. Its own pass, not part of process(): what a track sounds like has
    nothing to do with whether its lyrics were found, and the tracks that already carry lyrics want
    tags just as much as the pending ones."""
    with session() as db:
        t = db.get(Track, track_id)
        ratio = json.loads(t.match_info).get("ratio") if t.match_info else None
        # Parsed here rather than read off the row: Track.is_variant is only filled in by process(),
        # and a track can be tagged long before (or without) its lyrics ever being looked up.
        named = parse(t.tag_title, t.filename, t.tag_artist, t.folder).variant_word
        log.info("[tag] %s/%s%s", t.folder, t.filename, f" ({named})" if named else "")
        row = db.get(TrackTags, track_id)
        if row is None:
            row = TrackTags(track_id=track_id)
            db.add(row)
        fatal = None
        try:
            from rescale_tagger import tag  # heavy import, only when needed
            with _localized(t) as audio:
                r = tag(audio, named=named, ratio=ratio)
            row.genres, row.moods = json.dumps(r["genres"]), json.dumps(r["moods"])
            row.bpm, row.variant, row.speed = r["bpm"], r["variant"], r["speed"]
            row.model, row.error = r["model"], None
            log.info("  -> %s %s bpm %s", [g for g, _ in r["genres"]], r["bpm"], r["variant"] or "")
        except Exception as e:  # noqa: BLE001 - one bad track must not stop the batch
            log.exception("  -> tagging failed: %s", e)
            row.error = f"{type(e).__name__}: {e}"
            fatal = e if "cuda" in str(e).lower() else None  # a dead GPU fails every later track too
        db.commit()
        if fatal:
            raise RuntimeError(f"GPU is unusable in this process ({fatal}) - stopping. Restart and re-run `tag`.") from fatal
        return row


def select_untagged(lib: Library | None = None, contains: str | None = None, limit: int | None = None,
                    retag: bool = False) -> list[int]:
    """Tracks with no tags yet. A row that failed is not "tagged", so it comes back on the next run."""
    with session() as db:
        q = in_library(db.query(Track.id), lib).order_by(Track.folder, Track.filename)
        if not retag:
            q = q.filter(Track.id.notin_(select(TrackTags.track_id).where(TrackTags.error.is_(None))))
        if contains:
            q = q.filter(Track.path.contains(contains))
        if limit:
            q = q.limit(limit)
        return [i for (i,) in q]


def tags_of(track_id: int) -> dict | None:
    with session() as db:
        row = db.get(TrackTags, track_id)
        return row.as_dict() if row else None
