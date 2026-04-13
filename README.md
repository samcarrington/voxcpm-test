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
