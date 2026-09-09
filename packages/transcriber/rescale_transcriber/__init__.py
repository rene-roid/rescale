"""AI path: Demucs vocal separation -> WhisperX transcription + forced alignment -> LRC lines.

Everything runs locally; model weights land in data/models/ (TORCH_HOME / HF_HOME are pointed there
before torch is imported)."""
from __future__ import annotations

import hashlib
import logging
import zlib
from functools import cache, lru_cache
from pathlib import Path

from rescale_core import config, data_dir, point_model_caches_at_data_dir

point_model_caches_at_data_dir()  # must run before any torch / huggingface import


def device() -> str:
    d = config()["transcriber"]["device"]
    if d != "auto":
        return d
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_type() -> str:
    c = config()["transcriber"]["compute_type"]
    return c if c != "auto" else ("float16" if device() == "cuda" else "int8")


def free_vram(tag: str = "") -> None:
    """Hand torch's cached blocks back to the driver, and log what is left. Demucs (torch) and WhisperX
    (ctranslate2) allocate from separate pools, so torch must return blocks before ctranslate2 can use them.
    The log line is here because an OOM in ctranslate2 kills the process without saying how much was free."""
    if device() != "cuda":
        return
    import torch
    torch.cuda.empty_cache()
    if tag:
        free, total = torch.cuda.mem_get_info()
        logging.getLogger("rescale").info("  vram %s: %.2fG free of %.2fG", tag, free / 2**30, total / 2**30)


@cache
def _separator():
    from demucs.api import Separator
    return Separator(model=config()["transcriber"]["demucs_model"], device=device())


def separate_vocals(audio: Path) -> Path:
    """Isolated vocal stem as wav in data/stems/, keyed by path+mtime so re-runs are free."""
    st = audio.stat()
    key = hashlib.sha1(f"{audio}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()
    out = data_dir("stems") / f"{key}.wav"
    if out.exists():
        return out
    from demucs.api import save_audio
    sep = _separator()
    _, stems = sep.separate_audio_file(audio)
    tmp = out.with_suffix(".tmp.wav")
    save_audio(stems["vocals"], tmp, samplerate=sep.samplerate)
    tmp.rename(out)
    del stems
    free_vram()
    return out


@cache
def languages() -> list[str]:
    """The languages actually present in the library, validated against what whisperx can align.
    Whisper detects from the first 30 s of the vocal stem - usually an instrumental intro, which after
    demucs is near silence - and confidently returns Javanese or Latin for an English cover."""
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF as HF, DEFAULT_ALIGN_MODELS_TORCH as TORCH
    langs = config()["transcriber"]["languages"]
    bad = [x for x in langs if x not in HF and x not in TORCH]
    if bad:
        raise ValueError(f"transcriber.languages: whisperx has no alignment model for {bad}; "
                         f"pick from {sorted(set(HF) | set(TORCH))}")
    return langs


@cache
def _whisper():
    import whisperx
    cfg = config()["transcriber"]
    return whisperx.load_model(cfg["whisper_model"], device(), compute_type=compute_type(),
                               download_root=str(data_dir("models/whisper")))


@lru_cache(maxsize=1)  # one language at a time: a mixed library would otherwise pin a wav2vec2 per language
def _aligner(language: str):
    import whisperx
    return whisperx.load_align_model(language, device(), model_dir=str(data_dir("models/hf")))


@cache
def raw_transcribe(vocals: Path) -> dict:
    """Whisper only, no alignment: {'language': 'en', 'segments': [{start, end, text}]}. Cached per stem
    because the online-verification step and the AI path both need it."""
    import whisperx
    free_vram("before whisper")
    langs, batch = languages(), config()["transcriber"]["batch_size"]
    audio = whisperx.load_audio(str(vocals))
    # One language configured -> skip detection entirely (whisperx warns it costs inference time).
    result = _whisper().transcribe(audio, batch_size=batch, language=langs[0] if len(langs) == 1 else None)
    if result["language"] not in langs:
        # Misdetected: the text was decoded in the wrong language too, so redo it rather than just relabel.
        logging.getLogger("rescale").info("  detected %r, not in %s - re-transcribing as %r",
                                          result["language"], langs, langs[0])
        result = _whisper().transcribe(audio, batch_size=batch, language=langs[0])
    return {"language": result["language"], "segments": [{k: s[k] for k in ("start", "end", "text")} for s in result["segments"]]}


@cache
def transcribe(vocals: Path) -> dict:
    """raw_transcribe + wav2vec2 forced alignment: segments gain words: [{word, start, end, score}]."""
    import whisperx
    raw = raw_transcribe(vocals)
    audio = whisperx.load_audio(str(vocals))
    model, meta = _aligner(raw["language"])
    free_vram()  # only now is a swapped-out language's model unreferenced, so only now can its blocks go back
    aligned = whisperx.align(raw["segments"], model, meta, audio, device())
    free_vram()
    return {"language": raw["language"], "segments": aligned["segments"]}


def observed_lines(audio: Path) -> list[tuple[float, str]]:
    """[(start_seconds, text)] of what Whisper hears on the vocal stem, phrase-level, for verifying/fitting online lyrics."""
    res = transcribe(separate_vocals(audio))
    return to_lines(res["segments"], res["language"])


NO_SPACES = {"ja", "zh"}  # whisperx aligns these per character
# Whisper's favourite things to invent over instrumental sections.
HALLUCINATIONS = ("thanks for watching", "thank you for watching", "subscribe", "ご視聴ありがとう", "チャンネル登録", "字幕")
HALLUCINATED_LINES = {"thank you", "thanks", "you", "bye", "oh"}


def hallucinated(text: str) -> bool:
    """Repeated n-grams ("バラバラバラバラ…", "la la la la la…") compress far too well; stock phrases are on a denylist."""
    raw = text.encode()
    low = text.lower().strip(" .!?,")
    return (len(raw) > 24 and len(zlib.compress(raw)) / len(raw) < 0.45) or low in HALLUCINATED_LINES or any(h in low for h in HALLUCINATIONS)


def _line_break(prev: str | None, word: str) -> bool:
    """Whisper reproduces lyric formatting: a sentence end, or a Capitalised word after an unpunctuated one, starts a new line."""
    if prev is None:
        return False
    if prev[-1] in ".!?":
        return True
    return word[:1].isupper() and prev[-1] not in ",;:-" and not (word == "I" or word.startswith("I'"))


def to_lines(segments: list[dict], language: str = "en", max_words: int = 12, max_gap: float = 1.0) -> list[tuple[float, str]]:
    """Group aligned words into lyric lines: a new line at each segment, at a pause > max_gap, or after max_words."""
    sep = "" if language in NO_SPACES else " "
    max_words = max_words * 2 if language in NO_SPACES else max_words
    lines: list[tuple[float, str]] = []
    for seg in segments:
        words = seg.get("words") or []
        cur: list[str] = []
        start = seg.get("start")
        last_end = None
        for w in words:
            ws, we = w.get("start"), w.get("end")
            gap = (ws - last_end) if (ws is not None and last_end is not None) else 0
            if cur and (len(cur) >= max_words or gap > max_gap or (sep and _line_break(cur[-1], w["word"]))):
                lines.append((start, sep.join(cur)))
                cur, start = [], None
            if start is None and ws is not None:
                start = ws
            cur.append(w["word"])
            if we is not None:
                last_end = we
        if cur:
            if start is None:  # no aligned timestamps at all in this run -> attach to previous line time
                start = lines[-1][0] if lines else 0.0
            lines.append((start, sep.join(cur)))
    return [(round(float(t), 2), txt.strip()) for t, txt in lines if txt.strip() and not hallucinated(txt)]


def mean_score(segments: list[dict]) -> float:
    scores = [w["score"] for s in segments for w in s.get("words") or [] if "score" in w]
    return round(float(sum(scores) / len(scores)), 3) if scores else 0.0


def transcribe_to_lrc(audio: Path, title: str | None = None, artist: str | None = None) -> tuple[str, dict]:
    from rescale_rescaler import build
    vocals = separate_vocals(audio)
    res = transcribe(vocals)
    lines = to_lines(res["segments"], res["language"])
    info = {"language": res["language"], "score": mean_score(res["segments"]), "lines": len(lines), "stem": str(vocals)}
    return build(lines, title, artist), info
