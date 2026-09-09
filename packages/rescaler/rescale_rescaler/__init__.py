"""Pure LRC timestamp remapping. No audio I/O."""
from __future__ import annotations

import re

TS = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def parse_ts(m: re.Match) -> float:
    return int(m.group(1)) * 60 + float(m.group(2))


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(seconds, 60)
    return f"[{int(m):02d}:{s:05.2f}]"


def rescale(lrc: str, ratio: float, offset: float = 0.0) -> str:
    """t' = t * ratio + offset, applied to every timestamp on every line. Metadata lines pass through."""
    def fix(m: re.Match) -> str:
        return fmt_ts(parse_ts(m) * ratio + offset)
    return "\n".join(TS.sub(fix, line) for line in lrc.splitlines())


def lines(lrc: str) -> list[tuple[float, str]]:
    """[(seconds, text)] sorted, one entry per timestamp (lines with several tags are expanded)."""
    out = []
    for line in lrc.splitlines():
        stamps = TS.findall(line)
        if not stamps:
            continue
        text = TS.sub("", line).strip()
        out += [(int(m) * 60 + float(s), text) for m, s in stamps]
    return sorted(out)


def build(entries: list[tuple[float, str]], title: str | None = None, artist: str | None = None) -> str:
    head = [f"[{k}:{v}]" for k, v in (("ti", title), ("ar", artist), ("re", "Rescale")) if v]
    return "\n".join(head + [f"{fmt_ts(t)}{text}" for t, text in entries]) + "\n"


# ---- time-map fitting: LRC line times -> observed (transcribed) times ----------------------------------

def _norm(s: str) -> str:
    import re
    import unicodedata
    return re.sub(r"[^0-9a-z぀-ヿ一-鿿가-힯]+", "", unicodedata.normalize("NFKC", s).lower())


def fit(lrc_entries: list[tuple[float, str]], observed: list[tuple[float, str]], min_sim: float = 0.6):
    """Linear map t_obs ≈ ratio * t_lrc + offset, from LRC lines paired with transcribed lines.

    Pairing is a monotonic (in-order) alignment of the two line sequences, so a chorus repeated
    four times pairs with the right repetition. Ratio/offset come from Theil-Sen (median of pairwise
    slopes) so stray pairings don't matter. Returns (ratio, offset, n_pairs, median_residual_s) or None.
    Handles sped-up/slowed audio, trimmed intros and leading silence in one go."""
    import difflib
    import statistics
    lrc = [(t, _norm(x)) for t, x in lrc_entries]
    obs = [(t, _norm(x)) for t, x in observed]
    n, m = len(obs), len(lrc)
    if not n or not m:
        return None
    sim = [[difflib.SequenceMatcher(None, o, l).ratio() if len(o) >= 6 and l else 0.0 for _, l in lrc] for _, o in obs]
    skip = 0.02  # small cost per skipped line: prefer compact alignments, still allow a trimmed intro
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            gain = sim[i - 1][j - 1] - min_sim
            best[i][j] = max(best[i - 1][j] - skip, best[i][j - 1] - skip, best[i - 1][j - 1] + max(gain, -skip))
    pairs, i, j = [], n, m
    while i and j:
        if best[i][j] == best[i - 1][j - 1] + max(sim[i - 1][j - 1] - min_sim, -skip):
            if sim[i - 1][j - 1] >= min_sim:
                pairs.append((lrc[j - 1][0], obs[i - 1][0]))
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j] - skip:
            i -= 1
        else:
            j -= 1
    if len(pairs) < 4:
        return None
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for k, a in enumerate(pairs) for b in pairs[k + 1:] if abs(b[0] - a[0]) > 5]
    if not slopes:
        return None
    ratio = statistics.median(slopes)
    offset = statistics.median(t - ratio * l for l, t in pairs)
    resid = statistics.median(abs(t - (ratio * l + offset)) for l, t in pairs)
    return round(float(ratio), 4), round(float(offset), 2), len(pairs), round(float(resid), 2)
