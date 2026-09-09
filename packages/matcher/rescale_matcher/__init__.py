"""Filename/tag parsing + LRCLIB synced-lyrics lookup."""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from rescale_core import config, data_dir

LRCLIB = "https://lrclib.net/api"
UA = "Rescale/0.1 (https://github.com/yuuki/rescale)"

# Bracket pairs whose content is a hint (artist / source / flag), not part of the title.
BRACKETS = re.compile(r"[\(\[【「『（](.*?)[\)\]】」』）]")
NOISE = re.compile(r"\b(lyrics?|official (visualizer|video|audio|music video)|hd|hq|master)\b(-\d+)?", re.I)
FEAT = re.compile(r"\b(ft|feat|featuring)\.?\s+.*$", re.I)
# "black rover but its a swing arrangement by will stetson" / "credits song ... but im the final boss"
BUT = re.compile(r"\s+but\s+(it'?s|im|i'm|i)\b.*$", re.I)
LEAD_JUNK = re.compile(r"^[\W_]+|[\W_]+$")


@dataclass
class Parsed:
    title: str
    artist: str | None = None  # confident "Artist - Title" split
    hints: list[str] = field(default_factory=list)  # possible artists / sources pulled from brackets and " | " parts
    is_variant: bool = False
    is_cover: bool = False


def _has(words: list[str], s: str) -> bool:
    s = s.lower()
    return any(w in s for w in words)


def parse(title: str | None, filename: str, tag_artist: str | None = None, folder: str = "") -> Parsed:
    cfg = config()["matcher"]
    variant_words, cover_words = cfg["variant_words"], cfg["cover_words"]
    raw = (title or re.sub(r"\.[^.]+$", "", filename)).replace("_", " ")
    raw = unicodedata.normalize("NFKC", raw)
    flags = {"variant": any(_has(variant_words, x) for x in (raw, filename, folder)), "cover": _has(cover_words, raw)}
    hints: list[str] = []
    if BUT.search(raw):
        flags["cover"] = True
        raw = BUT.sub("", raw)

    def bracket(m: re.Match) -> str:
        inner = m.group(1).strip()
        if not inner or NOISE.fullmatch(inner) or _has(variant_words + cover_words, inner) or FEAT.match(inner):
            return " "
        hints.append(inner)
        return " "

    raw = BRACKETS.sub(bracket, raw)
    for w in variant_words:
        raw = re.sub(re.escape(w), " ", raw, flags=re.I)
    parts = [p.strip() for p in re.split(r"\s*\|+\s*|\s*→\s*", raw) if p.strip()]
    main, extra = parts[0] if parts else "", parts[1:]
    for p in extra:
        if not NOISE.fullmatch(p):
            hints.append(p)
    main = NOISE.sub(" ", main)
    main = FEAT.sub("", main)
    artist = None
    if " - " in main:
        left, right = main.split(" - ", 1)
        left, right = LEAD_JUNK.sub("", left), LEAD_JUNK.sub("", right)
        if left and right:
            artist, main = left, right
        else:
            main = right or left
    main = LEAD_JUNK.sub("", re.sub(r"\s{2,}", " ", main)).strip()
    if tag_artist:
        hints.append(tag_artist)
    return Parsed(main, artist, [h for h in hints if h], flags["variant"], flags["cover"])


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^0-9a-z぀-ヿ一-鿿가-힯]+", "", s)


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def _get(path: str, **params) -> list | dict | None:
    url = f"{LRCLIB}/{path}?{urllib.parse.urlencode({k: v for k, v in params.items() if v})}"
    cache = data_dir("cache/lrclib") / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read().decode()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                body = "null"
                break
            if e.code < 500 or attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))  # 503 / rate limit: 2, 4, 8, 16, 32 s
        except urllib.error.URLError:
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))
    cache.write_text(body)
    time.sleep(0.5)  # be polite to a free service
    return json.loads(body)


def search(p: Parsed) -> list[dict]:
    """LRCLIB candidates with synced lyrics, deduplicated, artist-scoped query first."""
    seen, out = set(), []
    queries = [dict(track_name=p.title, artist_name=p.artist)] if p.artist else []
    queries += [dict(track_name=p.title), dict(q=p.title)]
    for q in queries:
        for r in _get("search", **q) or []:
            if r.get("syncedLyrics") and r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        if out:  # first query that yields synced candidates wins; later ones are only fallbacks
            break
    return out


@dataclass
class Match:
    record: dict
    confidence: float
    title_sim: float
    duration_score: float
    ratio: float  # track_duration / original_duration (1.0 for a non-variant)
    lyric_sim: float | None = None  # transcript overlap, set by the pipeline when verification is on


def score(p: Parsed, r: dict, duration: float | None) -> Match:
    t = sim(p.title, r["trackName"])
    orig = r.get("duration") or 0
    ratio = (duration / orig) if duration and orig else 1.0
    if not duration or not orig:
        d = 0.5
    elif p.is_variant:
        # nightcore/sped up -> 0.72-0.92 typically; ~1.0 = untouched file in a nightcore folder; >1 = slowed.
        # Below 0.7 the rip is usually trimmed too, so duration alone can't place the lyrics.
        d = 1.0 if 0.7 <= ratio <= 1.03 else 0.8 if 1.03 < ratio <= 1.5 else 0.6 if 0.55 <= ratio < 0.7 else 0.2
    else:
        diff = abs(duration - orig)
        d = 1.0 if diff <= 3 else max(0.0, 1 - diff / 20)
        ratio = 1.0
    conf = t * (0.55 + 0.45 * d)
    names = [p.artist] if p.artist else p.hints
    if names and max(sim(n, r["artistName"]) for n in names) >= 0.8:
        conf = min(1.0, conf + 0.1)
    return Match(r, round(conf, 3), round(t, 3), d, ratio)


def candidates(p: Parsed, duration: float | None) -> list[Match]:
    """Scored LRCLIB candidates, best first. Same title from several artists with nothing that names
    the artist -> the top one is capped at the review threshold (title alone is a coin flip)."""
    matches = sorted((score(p, r, duration) for r in search(p)), key=lambda m: -m.confidence)
    if not matches:
        return []
    names = [p.artist] if p.artist else p.hints
    rivals = {m.record["artistName"] for m in matches if m.title_sim >= 0.9 and m.duration_score >= 0.8}
    named = any(sim(n, m.record["artistName"]) >= 0.8 for m in matches for n in names)
    if len(rivals) > 1 and not named:
        matches[0].confidence = min(matches[0].confidence, config()["matcher"]["review"])
    return matches


def best_match(p: Parsed, duration: float | None) -> Match | None:
    c = candidates(p, duration)
    return c[0] if c else None


def _grams(s: str, n: int = 4) -> set[str]:
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def lyric_similarity(transcript: str, plain_lyrics: str | None) -> float:
    """Share of the transcript's character 4-grams that occur in the candidate's lyrics (language agnostic)."""
    t, l = _grams(norm(transcript)), _grams(norm(plain_lyrics or ""))
    return round(len(t & l) / len(t), 3) if t else 0.0
