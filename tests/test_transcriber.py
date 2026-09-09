import pytest
from rescale_transcriber import hallucinated, mean_score, to_lines


def w(word, s, e, score=0.9):
    return {"word": word, "start": s, "end": e, "score": score}


def test_to_lines_splits_on_gap_and_length():
    seg = {"start": 1.0, "end": 9.0, "text": "", "words": [w("a", 1.0, 1.2), w("b", 1.3, 1.5), w("c", 3.0, 3.2), *[w(f"x{i}", 4 + i * 0.1, 4.05 + i * 0.1) for i in range(13)]]}
    lines = to_lines([seg], max_words=5, max_gap=1.0)
    assert lines[0] == (1.0, "a b")
    assert lines[1][1].startswith("c x0")
    assert all(len(t.split()) <= 5 for _, t in lines)


def test_unaligned_words_keep_text():
    seg = {"start": 2.0, "end": 3.0, "text": "", "words": [{"word": "1999"}, w("baby", 2.5, 2.9)]}
    assert to_lines([seg]) == [(2.0, "1999 baby")]  # falls back to the segment start
    assert mean_score([seg]) == 0.9


def test_japanese_joins_chars_and_drops_hallucinations():
    seg = {"start": 0, "end": 3, "text": "", "words": [w(c, i * 0.1, i * 0.1 + 0.05) for i, c in enumerate("恥ずかしいこと聞かないで")]}
    junk = {"start": 5, "end": 9, "text": "", "words": [w(c, 5 + i * 0.1, 5 + i * 0.1 + 0.05) for i, c in enumerate("バラ" * 12)]}
    assert to_lines([seg, junk], "ja") == [(0.0, "恥ずかしいこと聞かないで")]
    assert hallucinated("la la la la la la la la la la la la la")
    assert hallucinated("Thanks for watching!")
    assert not hallucinated("You were the shadow to my light")


def test_splits_on_capitalised_lyric_lines():
    text = "Baby, I'm wasted All I wanna do is drive home to you Baby, I'm faded"
    seg = {"start": 0, "end": 9, "text": text, "words": [w(x, i * 0.5, i * 0.5 + 0.3) for i, x in enumerate(text.split())]}
    assert [t for _, t in to_lines([seg], max_words=50)] == ["Baby, I'm wasted", "All I wanna do is drive home to you", "Baby, I'm faded"]


def test_languages_rejects_codes_whisperx_cannot_align(monkeypatch):
    import rescale_transcriber as rt
    cfg = {"transcriber": {"languages": ["en", "jw"]}}
    monkeypatch.setattr(rt, "config", lambda: cfg)
    rt.languages.cache_clear()
    with pytest.raises(ValueError, match="jw"):  # whisper detects Javanese on English covers; fail at config, not mid-run
        rt.languages()
    cfg["transcriber"]["languages"] = ["en", "ja"]
    rt.languages.cache_clear()
    assert rt.languages() == ["en", "ja"]
