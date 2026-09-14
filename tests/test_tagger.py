import numpy as np
import pytest
from rescale_tagger import SR, WINDOW, excerpts, fold_tempo, rank, variant

LABELS = ("nightcore", "j-pop", "techno")


def test_rank_orders_by_similarity_and_keeps_top_k():
    out = rank(np.array([0.30, 0.45, 0.31]), LABELS, top_k=2, min_score=0.0, temperature=20.0)
    assert [l for l, _ in out] == ["j-pop", "techno"]
    assert out[0][1] > out[1][1]


def test_rank_drops_labels_below_min_score():
    # One clear winner: the also-rans fall under the floor instead of being reported as tags.
    out = rank(np.array([0.60, 0.20, 0.20]), LABELS, top_k=3, min_score=0.12, temperature=20.0)
    assert [l for l, _ in out] == ["nightcore"]


def test_rank_scores_are_a_share_of_the_group():
    out = rank(np.array([0.4, 0.4, 0.4]), LABELS, top_k=3, min_score=0.0, temperature=20.0)
    assert sum(s for _, s in out) == pytest.approx(1.0, abs=0.01)  # scores are rounded to 3dp


def test_measured_ratio_beats_the_filename():
    # Sped up against the matched original -> nightcore, whatever the filename claimed.
    assert variant(174, [], None, 0.78, 150) == "nightcore"
    assert variant(120, [], None, 0.78, 150) == "sped-up"  # faster, but not nightcore tempo
    assert variant(90, [], "nightcore", 1.20, 150) == "slowed"  # measured slower than the original
    assert variant(174, ["nightcore"], "nightcore", 1.0, 150) is None  # measured as untouched: believe it


def test_named_variant_beats_the_audio_guess():
    assert variant(120, [], "nightcore", None, 150) == "nightcore"  # below the tempo gate, but named
    assert variant(174, [], "slowed", None, 150) == "slowed"
    assert variant(174, [], "daycore", None, 150) == "slowed"
    assert variant(174, [], "8d", None, 150) == "shifted"  # altered, without saying how


def test_audio_guess_only_without_a_ratio_or_a_name():
    assert variant(175, ["nightcore", "j-pop"], None, None, 150) == "nightcore"
    assert variant(140, ["nightcore"], None, None, 150) is None  # label without the tempo is not enough
    assert variant(140, ["j-pop"], None, None, 150) is None


def test_excerpts_span_the_track_and_pad_short_ones():
    size = SR * WINDOW
    long = np.arange(size * 10, dtype=np.float32)
    w = excerpts(long, 5)
    assert w.shape == (5, size)
    assert w[0][0] == 0 and w[-1][-1] == long[-1]  # first and last sample are both covered

    short = np.ones(size // 3, dtype=np.float32)
    w = excerpts(short, 5)
    assert w.shape == (1, size) and w[0][-1] == 0  # padded, not stretched


def test_fold_tempo_picks_the_octave():
    assert fold_tempo(77.1, 80, 175) == 154.2   # half-time reading of a nightcore rip
    assert fold_tempo(125.0, 80, 175) == 125.0  # already in range: left alone
    assert fold_tempo(360.0, 80, 175) == 90.0
    assert fold_tempo(0.0, 80, 175) == 0.0      # no beat found: doubling zero never reaches the floor
