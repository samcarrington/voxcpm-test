# VoxCPM2 TTS Test App

Small test app for trying **VoxCPM2** (`openbmb/VoxCPM2`) from Hugging Face.

It demonstrates:
- Voice design (text + style control)
- Voice cloning (reference audio)
- Streaming generation

## Requirements

- Python **>=3.10,<3.13** (project uses Python 3.12)
- macOS Apple Silicon is supported (uses MPS automatically)
- Disk space for model download (~10–15GB first run)

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Optional TTS engines

VoxCPM2 is installed by `requirements.txt`. Extra engines are optional and are
detected at runtime by the web UI and CLI.

```bash
# Kokoro preset-voice TTS
.venv/bin/pip install kokoro
# macOS: brew install espeak-ng

# Supertonic preset-voice TTS
.venv/bin/pip install supertonic

# NeuTTS Nano voice cloning
.venv/bin/pip install neutts
# Optional streaming/backend extras, if needed later:
# .venv/bin/pip install "neutts[all]" "neutts[llama]" pyaudio
```

Notes:

- Optional engines are not required for the app to start.
- Missing engines appear as `not_installed` in the web UI engine bar.
- Kokoro may need the system `espeak-ng` package for phonemizer/G2P support.
- NeuTTS clone mode requires both `--reference` audio and `--reference-text`.
- Supertonic and Kokoro do not support reference-audio cloning in this app.

## Quick start

Show environment info:

```bash
.venv/bin/python app.py --info
```

Run default voice design demo:

```bash
.venv/bin/python app.py
```

Run with custom text:

```bash
.venv/bin/python app.py --text "Welcome to our evening update."
```

Voice cloning:

```bash
.venv/bin/python app.py --clone --reference speaker.wav --text "This is a voice cloning test."
```

Streaming demo:

```bash
.venv/bin/python app.py --stream --text "Streaming synthesis test."
```

Select a runtime engine for CLI generation:

```bash
.venv/bin/python app.py --engine voxcpm --text "Hello from VoxCPM."
.venv/bin/python app.py --engine kokoro --text "Hello from Kokoro."
.venv/bin/python app.py --engine supertonic --text "Hello from Supertonic."
.venv/bin/python app.py --engine neutts --clone \
  --reference speaker.wav \
  --reference-text "Exact transcript of the reference audio" \
  --text "This is a NeuTTS clone test."
```

The CLI is single-engine only. It fails before model load when a selected engine
does not support the requested mode, for example Kokoro cloning or Supertonic
streaming.

Outputs are written to `output/`.

## Useful tuning flags

- `--cfg` guidance scale (e.g. 1.7–2.6)
- `--steps` inference timesteps for non-streaming
- `--stream-steps` timesteps for streaming
- `--attempts` app-level retries for broken outputs
- `--min-len` / `--max-len` decode length constraints
- `--retry-badcase-max-times`
- `--retry-badcase-ratio-threshold`
- `--disable-badcase-retry`
- `--seed` reproducibility
- `--no-normalize` disable text normalization

Example stability-focused run:

```bash
.venv/bin/python app.py \
  --text "Welcome to our evening update." \
  --steps 40 \
  --cfg 1.7 \
  --attempts 5 \
  --min-len 6 \
  --max-len 420 \
  --disable-badcase-retry \
  --seed 42
```

## Notes

- `VoxCPM.from_pretrained(...)` does **not** take a `device=` argument in this version.
- On Apple Silicon, VoxCPM2 auto-selects **MPS** internally when CUDA is unavailable.
- First run may be slow due to model download + warmup.

## Web UI

Local-only FastAPI server for interactive generation.

```bash
.venv/Scripts/python webapp.py --port 8000
# Open http://127.0.0.1:8000
```

Supports voice design, voice cloning (upload reference), and live streaming playback.

### API endpoints

- GET  /api/status
- GET  /api/info
- GET  /api/engines
- GET  /api/jobs
- GET  /api/jobs/{job_name}
- POST /api/jobs
- POST /api/generate
- WS   /api/generate/stream
- GET  /api/outputs
- GET  /api/uploads
- POST /api/uploads

### Constraints

Single user, localhost-only (binds 127.0.0.1). Reference uploads saved to `uploads/` (gitignored).
