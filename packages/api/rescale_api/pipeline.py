"""Orchestration: pick a path per track, run it, write the sidecar, record status in the DB."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rescale_core import Track, config, data_dir, session
from rescale_matcher import candidates, lyric_similarity, parse
from rescale_rescaler import build, fit, lines, rescale
from rescale_writer import write_lrc

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


def run_online(t: Track, p) -> tuple[str, str, float, dict] | None:
    cfg = config()["matcher"]
    cands = candidates(p, t.duration)[:3]
    if not cands:
        log.info("no LRCLIB candidates for %s", t.filename)
        return None
    info: dict = {}
    if cfg["verify"]:
        # Listen before trusting a title match: overlap between what Whisper hears and each candidate's lyrics.
        from rescale_transcriber import observed_lines
        obs = observed_lines(Path(t.path))
        heard = " ".join(x for _, x in obs)
        for m in cands:
            m.lyric_sim = lyric_similarity(heard, m.record["plainLyrics"])
        cands.sort(key=lambda m: (-(m.lyric_sim or 0), -m.confidence))
        m = cands[0]
        log.info("  candidates: %s", [(c.record["trackName"], c.record["artistName"], c.confidence, c.lyric_sim) for c in cands])
        if (m.lyric_sim or 0) < cfg["min_lyric_sim"]:
            return None
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
    info.update({"lrclib_id": r["id"], "track": r["trackName"], "artist": r["artistName"], "album": r.get("albumName"),
                 "orig_duration": r["duration"], "ratio": ratio, "offset": offset, "title_sim": m.title_sim, "lyric_sim": m.lyric_sim})
    return lrc, status, conf, info


def run_ai(t: Track, p) -> tuple[str, str, float, dict]:
    from rescale_transcriber import transcribe_to_lrc  # heavy import, only when needed
    lrc, info = transcribe_to_lrc(Path(t.path), p.title, p.artist or t.tag_artist)
    status = "ai-transcribed" if info["score"] >= config()["transcriber"]["review_below"] else "needs-review"
    return lrc, status, info["score"], info


def process(track_id: int) -> Track:
    with session() as db:
        t = db.get(Track, track_id)
        p = parse(t.tag_title, t.filename, t.tag_artist, t.folder)
        t.parsed_title, t.parsed_artist, t.is_variant, t.is_cover = p.title, p.artist, p.is_variant, p.is_cover
        path = decide(t)
        log.info("[%s] %s/%s  title=%r artist=%r variant=%s cover=%s", path, t.folder, t.filename, p.title, p.artist, p.is_variant, p.is_cover)
        try:
            result = None
            if path == "online":
                result = run_online(t, p)
                if result is None and t.path_pref == "auto":
                    path = "ai"
            if path == "ai":
                result = run_ai(t, p)
            if result is None:
                raise LookupError("no online lyrics match (path preference is online-only)")
            lrc, status, conf, info = result
            write_lrc(Path(t.path), lrc)
            t.lyrics, t.status, t.confidence, t.match_info, t.error = lrc, status, conf, json.dumps(info), None
            t.lrc_written_at = datetime.now(timezone.utc)
            log.info("  -> %s conf=%s %s", status, conf, info)
        except Exception as e:  # noqa: BLE001 - one bad track must not stop the batch
            log.exception("  -> failed: %s", e)
            t.status, t.error = "failed", f"{type(e).__name__}: {e}"
        t.path_used = path
        db.commit()
        return t


def select_ids(statuses=("pending",), contains: str | None = None, limit: int | None = None) -> list[int]:
    with session() as db:
        q = db.query(Track.id).filter(Track.status.in_(statuses)).order_by(Track.folder, Track.filename)
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


def counts() -> dict:
    with session() as db:
        return {s: db.query(Track).filter(Track.status == s).count() for s in ("pending", "matched-online", "ai-transcribed", "needs-review", "failed")}
