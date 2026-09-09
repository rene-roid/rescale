# Rescale

Scans a music library (read-only) and writes a time-synced `<track>.lrc` sidecar next to every
track it can, for Navidrome to pick up as external synced lyrics.

Two paths per track, chosen automatically and overridable per track:

- **online**: parse the original title/artist out of the filename and tags ("Nightcore - X (Lyrics) | Uploader"),
  look up synced lyrics on [LRCLIB](https://lrclib.net), and for nightcore / sped-up / slowed variants rescale
  every timestamp by `track_duration / original_duration`.
- **ai**: Demucs vocal separation, WhisperX transcription and forced alignment, all local
  (GPU if present). Used for covers, remixes, anything with no online match, and anything you point at it.

All generated data (SQLite db, LRCLIB cache, model weights, vocal stems, logs) lives in `data/` inside
this repo. The library tree is never touched except for the `.lrc` sidecars, which are written atomically.

## Setup

```sh
uv sync                                  # ~5 GB: torch cu128 + whisperx + demucs
uv run rescale scan --report             # dry run: extensions, tag quality, sample rows
uv run rescale scan                      # populate data/db/rescale.sqlite3
uv run rescale run --limit 5             # process 5 pending tracks, write .lrc sidecars
uv run rescale run                       # everything pending (model weights download on first AI track)
uv run rescale serve                     # http://localhost:8765  UI + JSON API
```

Other commands:

```sh
uv run rescale status                              # counts per status
uv run rescale run --retry-failed --redo-review    # widen what "run" picks up
uv run rescale run --filter "Will Stetson"         # only paths containing a substring
uv run rescale prefer ai --filter "nightcore is B)" # force a path for matching tracks, mark them pending
uv run pytest
```

Config: `config/default.toml`. Any key can be overridden with `RESCALE_<SECTION>_<KEY>` env vars
(`RESCALE_LIBRARY_ROOT=/music`, `RESCALE_TRANSCRIBER_DEVICE=cpu`, ...).

Statuses: `pending`, `matched-online`, `ai-transcribed`, `needs-review` (written, but confidence below
the accept threshold), `failed`. Re-runs skip everything that isn't pending.

## Docker

```sh
docker compose up -d --build
```

Edit the library path in `compose.yaml`. Drop the `deploy.resources` block on a machine without an NVIDIA
GPU and set `RESCALE_TRANSCRIBER_DEVICE=cpu` (expect roughly real-time or slower per track for the AI path).

## Layout

uv workspace, one package per stage: `core` (config + db), `scanner`, `matcher` (parser + LRCLIB),
`rescaler` (pure timestamp math), `transcriber` (Demucs + WhisperX), `writer` (atomic sidecar),
`api` (pipeline, CLI, FastAPI + `frontend/`).
