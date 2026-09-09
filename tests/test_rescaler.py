from rescale_rescaler import build, fit, fmt_ts, lines, rescale

FIXTURE = """[ti:Faded]
[ar:Alan Walker]
[00:12.50]You were the shadow to my light
[00:15.00][01:30.00]Did you feel us
[01:02.345]Another star
"""


def test_roundtrip_identity():
    assert rescale(FIXTURE, 1.0) == FIXTURE.rstrip("\n").replace("[01:02.345]", "[01:02.34]")  # timestamps normalised to 2 decimals


def test_rescale_ratio_and_offset():
    out = rescale(FIXTURE, 0.8, offset=0.5)
    assert "[00:10.50]You were" in out  # 12.5*0.8+0.5
    assert "[00:12.50][01:12.50]Did you" in out  # 15*0.8+.5 ; 90*.8+.5
    assert "[00:50.38]Another" in out  # 62.345*0.8+0.5 = 50.376
    assert out.startswith("[ti:Faded]\n[ar:Alan Walker]")


def test_lines_expand_and_sort():
    assert lines(FIXTURE) == [(12.5, "You were the shadow to my light"), (15.0, "Did you feel us"), (62.345, "Another star"), (90.0, "Did you feel us")]


def test_fmt_ts():
    assert fmt_ts(0) == "[00:00.00]"
    assert fmt_ts(-1) == "[00:00.00]"
    assert fmt_ts(3599.999) == "[59:60.00]" or fmt_ts(3599.999) == "[60:00.00]"
    assert fmt_ts(125.678) == "[02:05.68]"


def test_build():
    assert build([(1.0, "a"), (2.5, "b")], "T", None) == "[ti:T]\n[re:Rescale]\n[00:01.00]a\n[00:02.50]b\n"


def test_fit_recovers_speed_and_offset():
    lrc = [(10.0 * i, f"line number {i} of the song lyrics here") for i in range(12)]
    observed = [(10.0 * i * 0.8 + 1.5, f"line number {i} of the song lyrics here") for i in range(12) if i % 3]
    observed.append((50.0, "completely unrelated words that match nothing"))
    ratio, offset, n, resid = fit(lrc, observed)
    assert abs(ratio - 0.8) < 0.01 and abs(offset - 1.5) < 0.1 and n == 8 and resid < 0.1


def test_fit_needs_agreement():
    assert fit([(0, "hello world today"), (10, "goodbye moon tonight")], [(0, "hello world today")]) is None


def test_fit_repeated_chorus_trimmed_intro():
    chorus = ["baby i'm wasted", "all i wanna do is drive home to you", "baby i'm faded", "all i wanna do is take you downtown"]
    lrc = [(45.8 + 4.0 * i, chorus[i % 4]) for i in range(16)]   # original: 45 s intro, chorus x4
    obs = [(1.6 + 3.2 * i, chorus[i % 4]) for i in range(16)]    # nightcore: intro cut, 0.8 speed
    obs[5] = (obs[5][0], "mumble")                                 # one line Whisper got wrong
    ratio, offset, n, resid = fit(lrc, obs)
    assert abs(ratio - 0.8) < 0.01 and abs(offset - (1.6 - 45.8 * 0.8)) < 0.2 and n >= 14 and resid < 0.1
