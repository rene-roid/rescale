<div align="center">

<img src="frontend/icon.png" width="96" alt="Rescale" />

# Rescale

**Every track in your library gets synced lyrics. No exceptions.**

Point it at your music folder — it scans, finds synced lyrics online, rescales the timestamps for
nightcore and sped-up edits, and transcribes the rest locally with Demucs + WhisperX. Out comes a
`.lrc` sidecar next to every file, ready for Navidrome.

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/packaging-uv-de5fe9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/ML-WhisperX%20%2B%20Demucs-ee4c2c?logo=pytorch&logoColor=white)](https://github.com/m-bain/whisperX)
[![SQLite](https://img.shields.io/badge/db-SQLite-003b57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## Screenshots

<table>
<tr>
<td colspan="2">

**Library** — every track, its status, confidence, and which path picked it up
<img src="docs/screenshots/dashboard.png" alt="Track list" />

</td>
</tr>
<tr>
<td width="50%">

**Online path** — LRCLIB match, measured speed ratio and offset, synced lyrics
<img src="docs/screenshots/player.png" alt="Online match detail" />

</td>
<td width="50%">

**AI path** — Whisper language, alignment score, transcribed and force-aligned lyrics
<img src="docs/screenshots/ai-path.png" alt="AI transcription detail" />

</td>
</tr>
</table>

## Why "Rescale"

A synced lyric file is a list of timestamps, and those timestamps only mean something for the exact audio they were built against. Speed up a track for a nightcore edit, slow it down, trim the intro — the words are unchanged but every timestamp now points at the wrong second. Rescale's job is exactly what the name says: measure how a track's timeline diverges from the original and stretch or compress the timestamps to fit it, whether that's a sped-up edit, a slowed one, or anything else whose timing doesn't match the source.

## How it works

```
scan ──> decide ──┬── online ──> LRCLIB lookup ──> verify against the audio ──> rescale ──┐
                  │                                                                       ├──> <track>.lrc
                  └── ai ──────> Demucs vocals ──> WhisperX ──> forced alignment ─────────┘
```

Every track picks one of two paths automatically, and you can override it per track:

- **online** — parses the real title and artist out of YouTube-rip filenames
  (`Nightcore - X (Lyrics) | Uploader`), looks the song up on [LRCLIB](https://lrclib.net), then
  **measures the actual speed change and lead-in trim from your copy of the audio** and rescales
  every timestamp to match. A 1.25× nightcore edit lines up to the word.
- **ai** — Demucs separates the vocal stem, WhisperX transcribes and force-aligns it, entirely
  local, on GPU if you have one. Used for covers, remixes, obscure tracks with no online match, and
  anything you point at it by hand.

The online path doesn't trust a title match: it runs a quick transcript of the vocal stem and scores
every candidate on how much of it it actually heard. Wrong song, heavily cut edit, nonsense timing
fit — all of it falls through to the AI path instead of writing garbage.

## Features

- 🎯 **Two paths, chosen per track** — online lookup where it's right, local AI where it isn't, `prefer` to override
- ⏱ **Timestamp rescaling** — nightcore, sped up, slowed, daycore: speed *and* offset measured from the audio, not guessed from a duration ratio
- 🎧 **Audio verification** — every online candidate is scored against a Whisper transcript of the vocal stem before it's accepted
- 🧠 **Local transcription** — Demucs `htdemucs` + WhisperX `large-v3` with word-level forced alignment, no API keys, nothing leaves the box
- 🔍 **Filename parsing** — strips `(Lyrics)`, `[Official Video]`, uploader suffixes and the rest of the YouTube-rip noise
- 📊 **Confidence gating** — low-confidence results land in `needs-review` instead of silently shipping wrong lyrics
- 🖥 **Web UI** — browse by status, filter by path, play a track with its lyrics highlighting live, click a line to seek, reprocess in one click
- 🔒 **Read-only library** — the *only* thing ever written into your music tree is the `.lrc` sidecar, atomically
- 🐳 **Docker + GPU** — CUDA 12.8 image, or drop the GPU block and run on CPU
- 📦 **All state in `data/`** — db, LRCLIB cache, model weights, vocal stems, logs, one folder, delete it to start over

## Tech Stack

| Layer | Choice |
|---|---|
| Runtime & packaging | Python 3.12, [uv](https://docs.astral.sh/uv/) workspace |
| Backend | FastAPI + SQLModel, SQLite |
| Frontend | One `index.html`, no build step |
| Lyrics source | [LRCLIB](https://lrclib.net) |
| ML | [WhisperX](https://github.com/m-bain/whisperX) (`large-v3`) + [Demucs](https://github.com/adefossez/demucs) (`htdemucs`), torch cu128 |
| Audio | FFmpeg |

## Quick Start

```sh
uv sync                       # ~5 GB: torch cu128 + whisperx + demucs
uv run rescale scan --report  # dry run: extensions, tag quality, sample rows
uv run rescale scan           # populate data/db/rescale.sqlite3
uv run rescale run --limit 5  # process 5 tracks, write .lrc sidecars
uv run rescale run            # everything pending (models download on the first AI track)
uv run rescale serve          # http://localhost:8765
```

Set your library path in `config/default.toml` or `RESCALE_LIBRARY_ROOT=/music` before scanning.

## Commands

```sh
uv run rescale status                               # counts per status
uv run rescale run --retry-failed --redo-review     # widen what "run" picks up
uv run rescale run --filter "Will Stetson"          # only paths containing a substring
uv run rescale prefer ai --filter "nightcore is B)" # force a path, mark those tracks pending
uv run pytest
```

**Statuses:** `pending` → `matched-online` | `ai-transcribed` | `needs-review` (written, but below
the accept threshold) | `failed`. Re-runs skip everything that isn't pending.

**Config:** `config/default.toml`. Any key overrides via `RESCALE_<SECTION>_<KEY>` —
`RESCALE_LIBRARY_ROOT=/music`, `RESCALE_TRANSCRIBER_DEVICE=cpu`,
`RESCALE_TRANSCRIBER_BATCH_SIZE=4` if WhisperX OOMs next to Demucs.

## API

| Method | Route | Does |
|---|---|---|
| `GET` | `/api/stats` | counts per status |
| `GET` | `/api/tracks?status=&q=` | list, filterable by status and path substring |
| `GET` | `/api/tracks/{id}` | one track with its lyrics |
| `GET` | `/api/tracks/{id}/audio` | stream the file, for the UI player |
| `POST` | `/api/tracks/{id}/reprocess?path=auto\|online\|ai` | requeue one track on a given path |
| `POST` | `/api/run?status=pending\|failed\|needs-review` | kick the background worker |
| `POST` | `/api/scan` | rescan the library |

## Docker

```sh
docker compose up -d --build
```

Edit the library path in `compose.yaml`. On a machine without an NVIDIA GPU, drop the
`deploy.resources` block and set `RESCALE_TRANSCRIBER_DEVICE=cpu` — expect roughly real-time or
slower per track on the AI path.

## Project Structure

One uv workspace package per pipeline stage, ~1100 lines total.

```
rescale/
├── packages/
│   ├── core/          # config loading + SQLModel schema + session
│   ├── scanner/       # read-only library walk, tag reading, upsert
│   ├── matcher/       # filename/tag parsing, LRCLIB search + cache, similarity scoring
│   ├── rescaler/      # pure timestamp math: parse, fit speed/offset, rescale, rebuild
│   ├── transcriber/   # Demucs stem separation, WhisperX transcribe + align
│   ├── writer/        # atomic .lrc sidecar write
│   └── api/           # pipeline orchestration, CLI, FastAPI app
├── frontend/          # index.html: track list, player, live-highlighting lyrics
├── config/            # default.toml
├── tests/
└── data/              # db, caches, model weights, stems, logs (gitignored)
```
