import pytest
from rescale_matcher import parse, sim

CASES = [  # (tag title or filename, tag artist) -> (title, artist, hints ⊇, variant, cover)
    (("Nightcore - Alone, Pt. II (Alan Walker & Ava Max) - (Lyrics)", "Syrex"), ("Alone, Pt. II", None, ["Alan Walker & Ava Max"], True, False)),
    (("Nightcore - Astronaut In The Ocean (Lyrics) | Masked Wolf", "Chino"), ("Astronaut In The Ocean", None, ["Masked Wolf"], True, False)),
    (("Nightcore - Heart Attack (Switching Vocals)", "Zen - Kun"), ("Heart Attack", None, [], True, True)),
    (("Nightcore - E.T (Rock Version) || Lyrics", "x"), ("E.T", None, [], True, True)),
    (("Nightcore → Heavyweight", "Shiko"), ("Heavyweight", None, [], True, False)),
    (("✧Nightcore - We're Just Friends (lyrics)", "Nightcore Wolfie"), ("We're Just Friends", None, [], True, False)),
    (("Nightcore - The Labyrinth (NIVIRO) - (Lyrics)", "Syrex"), ("The Labyrinth", None, ["NIVIRO"], True, False)),
    (("Defqwop - Heart Afire (ft. Strix)", "Diversity"), ("Heart Afire", "Defqwop", [], True, False)),  # variant flag comes from the folder name
    (("Marshmello x Imanbek - Too Much ft. Usher", "amazing."), ("Too Much", "Marshmello x Imanbek", [], False, False)),
    (("Mixed Nuts (English Cover)【 Will Stetson 】「SPY×FAMILY OP」", "Will Stetson"), ("Mixed Nuts", None, ["Will Stetson", "SPY×FAMILY OP"], False, True)),
    (("black rover but its a swing arrangement by will stetson", "Will Stetson"), ("black rover", None, [], False, True)),
    (("gurenge but its a swing arrangement by will stetson (demon slayer op)", "Will Stetson"), ("gurenge", None, [], False, True)),
    (("Michael Bublé - End of May (Will Stetson Cover)", "Will Stetson"), ("End of May", "Michael Bublé", [], False, True)),
    (("Credits song for my death but it's played on a broken fax machine", "Camel"), ("Credits song for my death", None, [], False, True)),
    (("credits song for my death (flydum's breakcore remix)", "flydum"), ("credits song for my death", None, [], False, True)),
    (("Nightcore -  Whenever, Wherever (Lyrics)", "Shiko"), ("Whenever, Wherever", None, [], True, False)),
    ((None, None), ("TRIPLE BAKA", None, [], False, False)),
]


@pytest.mark.parametrize("inp,exp", CASES)
def test_parse(inp, exp):
    title, artist = inp
    filename = "TRIPLE BAKA master-002.wav" if title is None else "x.mp3"
    folder = "nightcore is B)" if title and title.startswith("Defqwop") else "Will Stetson"
    p = parse(title, filename, artist, folder)
    e_title, e_artist, e_hints, e_variant, e_cover = exp
    assert p.title == e_title
    assert p.artist == e_artist
    for h in e_hints:
        assert h in p.hints
    assert p.is_variant == e_variant
    assert p.is_cover == e_cover


def test_sim():
    assert sim("Build A B*tch", "Build a Bitch") > 0.85
    assert sim("Closer", "Closer") == 1.0
    assert sim("Faded", "Fade") < 0.95


def test_lyric_similarity_discriminates_songs():
    from rescale_matcher import lyric_similarity
    zhu = "Baby I'm wasted, all I wanna do is drive home to you. Baby I'm faded, all I wanna do is take you downtown"
    alan = "You were the shadow to my light, did you feel us? Another star, you fade away. Afraid our aim is out of sight"
    heard = "baby i'm wasted all i wanna do is drive home to you baby i'm faded"
    assert lyric_similarity(heard, zhu) > 0.8
    assert lyric_similarity(heard, alan) < 0.25
