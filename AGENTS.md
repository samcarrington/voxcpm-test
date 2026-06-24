# VoxCPM2 TTS Test App — Agent Notes

## Structure

- `core.py` — shared logic: model loading, generation, WAV output, uploads, `ModelState` (lazy init + async lock), `streaming_adapter`. **The single source of truth** for all TTS operations.
- `app.py` — CLI front-end. Imports from `core`, adds argparse + demo orchestration (voice design / cloning / streaming). Flat by design; don't split unless executing web-ui plan tasks.
- `webapp.py` — FastAPI server (localhost:8000). Shares `core.py` entirely. Mounts `/outputs` as static, serves `web/` SPA if present. WebSocket at `/api/generate/stream` sends raw audio bytes + progress frames.
- `web/` — vanilla HTML/CSS/JS SPA (no framework).
- `example.py` — **stale**. Imports non-existent `VoxCPM2` class. Correct: `VoxCPM` from `voxcpm`.
- `PRODUCT.md` — design intent for web UI.
- `VOXCPM2_RESEARCH.md` — upstream notes; not a spec.
- `.weave/plans/` — pre-written plans. Read before generating new work to avoid duplicating.

## Environment

- **Python 3.12 only** (package constraint `<3.13`). Use the venv: `.venv\Scripts\python` (Windows) or `.venv/bin/python` (macOS/Linux).
- `requirements.txt` — 8 direct entries: `voxcpm, soundfile, numpy, fastapi, uvicorn[standard], websockets, python-multipart, pydantic`. No lockfile, no test/lint config.
- First run downloads ~10–15 GB from HuggingFace. Use `HF_ENDPOINT=https://hf-mirror.com` if blocked.

## Running

```bash
# CLI
.venv\Scripts\python app.py                          # voice design (default)
.venv\Scripts\python app.py --text "Hello"
.venv\Scripts\python app.py --clone -r ref.wav --text "Clone"
.venv\Scripts\python app.py --stream
.venv\Scripts\python app.py --info

# Web server (localhost only)
.venv\Scripts\python webapp.py --port 8000

# Stability-focused run
.venv\Scripts\python app.py --text "..." --steps 40 --cfg 1.7 --seed 42 --disable-badcase-retry
```

Outputs → `output/` (gitignored). Uploads → `uploads/` (gitignored).

## Device / hardware quirks

- `core.detect_device()` picks CUDA > MPS > CPU. `--info` prints full diagnostics.
- `VoxCPM.from_pretrained(...)` does **not** accept `device=`. Device placement is internal.
- `torch.compile` (`optimize=True`) is only enabled on **CUDA**. Skip on MPS/CPU.
- If NVIDIA GPU detected but PyTorch is CPU-built: `--info` shows install fix (cu128 wheel).

## CLI flags (core ones)

| Flag | Default | Notes |
|------|---------|-------|
| `--cfg` | 2.0 | Guidance. Lower (1.5–1.7) = more stable for long text. |
| `--steps` | 20 | Non-streaming diffusion steps. |
| `--stream-steps` | 12 | Streaming diffusion steps. |
| `--attempts` | 2 | App-level retries on bad waveform. |
| `--seed` | — | `torch.manual_seed` for reproducibility. |
| `--no-normalize` | off | Skip text normalization (numbers, dates). |
| `--denoiser` | off | ZipEnhancer denoiser (slower startup). |
| `--disable-badcase-retry` | off | Disable VoxCPM internal badcase retry loop. |

## What to avoid

- Don't rewrite `app.py` to split logic unless executing web-ui plan tasks. Intentionally flat.
- Don't run tests/linters — none configured.
- Don't assume `VoxCPM2` class exists. The package exports `VoxCPM`, and `from_pretrained()` takes no `device=` kwarg.
- Don't commit `output/`, `uploads/`, `.venv/`, `__pycache__/`.
- When both `app.py` and `webapp.py` call into `core.py`, the source of truth is `core.py` — changes should land there first.
