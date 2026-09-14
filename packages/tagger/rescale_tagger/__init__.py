"""Zero-shot audio tagging: CLAP text-audio similarity for genre/mood, librosa for tempo.

The candidate labels in [tagger] *are* the taxonomy - CLAP scores the audio against whatever
strings you give it, so a subgenre nobody trained a classifier on works the same as "pop".
Nothing here touches the DB or the library; the pipeline owns both."""
from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

import numpy as np
from rescale_core import config, point_model_caches_at_data_dir

point_model_caches_at_data_dir()  # must run before any torch / huggingface import

SR = 48_000   # CLAP's sample rate
WINDOW = 10   # seconds per excerpt: CLAP's native input length
GENRE_PROMPT = "{} music"
MOOD_PROMPT = "a {} sounding song"
# What a variant word in the filename means. Anything else known to the matcher ("8d", "boosted")
# says the file was altered without saying how, which is what "shifted" is for.
NAMED_VARIANT = {"nightcore": "nightcore", "sped up": "sped-up", "speed up": "sped-up",
                 "spedup": "sped-up", "slowed": "slowed", "daycore": "slowed"}


def device() -> str:
    d = config()["tagger"]["device"]
    if d != "auto":
        return d
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def decode(path: Path) -> np.ndarray:
    """48 kHz mono float32. One ffmpeg call decodes every format in the library the same way -
    libsndfile has gaps (aac), and both CLAP and the tempo tracker want the same array."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                       capture_output=True)
    if r.returncode:  # the reason is stored on the row, so it has to say what ffmpeg actually complained about
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {r.stderr.decode(errors='replace').strip()[:300]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()  # frombuffer is read-only; librosa wants writable


def excerpts(y: np.ndarray, n: int) -> np.ndarray:
    """n evenly spaced WINDOW-second excerpts, first to last sample. A remix that opens acoustic and
    drops into techno is two different sounds, so the tags come from across the track, not its start."""
    size = SR * WINDOW
    if len(y) <= size:
        return np.pad(y, (0, size - len(y)))[None, :]
    starts = np.linspace(0, len(y) - size, max(1, n)).astype(int)
    return np.stack([y[s:s + size] for s in starts])


@cache
def _model():
    from transformers import ClapModel, ClapProcessor
    name = config()["tagger"]["model"]
    return ClapModel.from_pretrained(name).to(device()).eval(), ClapProcessor.from_pretrained(name)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def audio_embedding(y: np.ndarray, windows: int) -> np.ndarray:
    """One unit vector for the whole track: mean of its excerpt embeddings."""
    import torch
    model, proc = _model()
    inputs = proc(audio=list(excerpts(y, windows)), sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        e = model.get_audio_features(**{k: v.to(device()) for k, v in inputs.items()})
    return _unit(_unit(e.float().cpu().numpy()).mean(0))


@cache
def text_embedding(labels: tuple[str, ...], prompt: str) -> np.ndarray:
    """Label vectors, cached: the taxonomy is the same for every track in a run."""
    import torch
    model, proc = _model()
    inputs = proc(text=[prompt.format(l) for l in labels], return_tensors="pt", padding=True)
    with torch.no_grad():
        e = model.get_text_features(**{k: v.to(device()) for k, v in inputs.items()})
    return _unit(e.float().cpu().numpy())


def rank(sims: np.ndarray, labels: tuple[str, ...], top_k: int, min_score: float,
         temperature: float) -> list[list]:
    """Best labels as a share of the group's probability mass. CLAP similarities are only meaningful
    against each other, never as absolute numbers, so scores are softmaxed within the group and
    `temperature` decides how sharp that is - raise it for one confident label, lower it to see more."""
    p = np.exp((sims - sims.max()) * temperature)
    p /= p.sum()
    order = np.argsort(-p)[:top_k]
    return [[labels[i], round(float(p[i]), 3)] for i in order if p[i] >= min_score]


def score(emb: np.ndarray, labels: tuple[str, ...], prompt: str) -> list[list]:
    cfg = config()["tagger"]
    sims = text_embedding(labels, prompt) @ emb
    return rank(sims, labels, cfg["top_k"], cfg["min_score"], cfg["temperature"])


def fold_tempo(tempo: float, low: float, high: float) -> float:
    """Tempo into [low, high] by doubling or halving.
    ponytail: half-time vs double-time is not decidable from audio - a half-time trap beat and its
    double-time reading are both true - so the octave is a convention from config, not a measurement."""
    while 0 < tempo < low:  # 0 = no beat found; doubling it forever would not help
        tempo *= 2
    while tempo > high:
        tempo /= 2
    return round(tempo, 1)


def bpm(y: np.ndarray, low: float, high: float) -> float:
    import librosa
    return fold_tempo(float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=SR)[0])[0]), low, high)


def variant(tempo: float, genres: list[str], named: str | None, ratio: float | None,
            nightcore_bpm: float) -> str | None:
    """nightcore | sped-up | slowed | shifted | None.

    `ratio` is track duration / original duration, measured against the matched original by the
    lyrics pipeline - when it exists it is the answer, because it is the only evidence here that
    compares the file to the song it came from. `named` is the variant word the filename used, which
    is weaker but still someone stating what they made.
    ponytail: with neither, this is a guess. A pitch shift is not provable from one recording alone,
    so tempo + the 'nightcore' label is the ceiling."""
    if ratio:
        if ratio < 0.95:
            return "nightcore" if tempo >= nightcore_bpm else "sped-up"
        return "slowed" if ratio > 1.05 else None
    if named:
        return NAMED_VARIANT.get(named, "shifted")
    return "nightcore" if "nightcore" in genres and tempo >= nightcore_bpm else None


def tag(path: Path, named: str | None = None, ratio: float | None = None) -> dict:
    """Tags for one audio file. `named` / `ratio` come from the lyrics pipeline - the variant word the
    filename used, and what the timing fit measured against the matched original."""
    cfg = config()["tagger"]
    y = decode(Path(path))
    emb = audio_embedding(y, cfg["windows"])
    genres = score(emb, tuple(cfg["genres"]), GENRE_PROMPT)
    tempo = bpm(y, *cfg["bpm_range"])
    return {
        "genres": genres,
        "moods": score(emb, tuple(cfg["moods"]), MOOD_PROMPT),
        "bpm": tempo,
        "variant": variant(tempo, [g for g, _ in genres], named, ratio, cfg["nightcore_bpm"]),
        "speed": round(1 / ratio, 3) if ratio else None,  # ratio is track/original, speed is its inverse
        "model": cfg["model"],
    }
