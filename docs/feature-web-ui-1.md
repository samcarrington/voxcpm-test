---
goal: Add a local-only FastAPI web front-end to the VoxCPM2 audio generation CLI app while preserving CLI behavior
version: 1.0
date_created: 2026-06-19
last_updated: 2026-06-19
owner: voxcpm-test maintainer
status: 'Planned'
tags: ['feature', 'architecture', 'refactor', 'web-ui', 'fastapi', 'tts']
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan specifies the implementation of a local-only web front-end for the existing VoxCPM2 text-to-speech test app. The current `app.py` CLI supports three generation modes (voice design, voice cloning, streaming) but offers no interactive UI. We will:

1. Refactor reusable TTS logic out of `app.py` into a new `core.py` module.
2. Preserve `app.py` as a thin CLI wrapper over `core.py` (backward compatible flags & outputs).
3. Add a new `webapp.py` FastAPI server exposing JSON REST routes, a WebSocket route for live streaming, and a static SPA served from `web/`.
4. Ship a vanilla (no-build) HTML/JS/CSS SPA that supports both final-WAV generation and live-chunk playback.

Deployment context: single user, run locally, bind `127.0.0.1` only. Source of truth for VoxCPM2 API surface is `docs/research/VOXCPM2_RESEARCH.md`.

## 1. Requirements & Constraints

- **REQ-001**: Web UI MUST support voice-design mode (text + optional voice description, no reference audio) producing one final WAV.
- **REQ-002**: Web UI MUST support voice-clone mode (text + reference audio upload) producing one final WAV.
- **REQ-003**: Web UI MUST support live-streaming mode that begins audio playback in the browser BEFORE server-side synthesis of the full utterance completes, using the same `model.generate_streaming()` generator that `demo_streaming()` uses in `app.py:409-452`.
- **REQ-004**: User MUST be able to choose, per request, between (a) "final WAV only" generation via `model.generate()` and (b) "live streaming" generation via `model.generate_streaming()`. Both options must exist as distinct actions in the UI.
- **REQ-005**: Generation request parameters MUST be limited to this common subset, applied to both final and streaming paths where applicable: `text` (required), `voice_description` (optional, wrapped as `(description)text` before passing to VoxCPM), `reference_wav_path` (optional; presence triggers clone mode), `cfg_value` (float 1.0–3.0, default 2.0), `inference_timesteps` (int 4–30, default 20 for final, 12 for streaming), `normalize` (bool, default true), `attempts` (int, default 2; final-only), `seed` (optional int).
- **REQ-006**: All generated WAV files MUST be persisted under `output/` and removed from git tracking per existing `.gitignore` entry `output/`.
- **REQ-007**: Uploaded reference audio MUST be saved under a new `uploads/` directory; previously-uploaded files MUST be listable and re-selectable within a session via a `/api/uploads` endpoint. Directory MUST be added to `.gitignore`.
- **REQ-008**: Existing `app.py` CLI flags, defaults, and console output (per `README.md` lines 24-66) MUST remain functionally unchanged. `python app.py --info`, `python app.py --text="..."`, `python app.py --clone --reference speaker.wav ...`, and `python app.py --stream ...` MUST all continue to work identically post-refactor.
- **REQ-009**: Only ONE global `VoxCPM` model instance MUST exist per `webapp.py` process lifetime. It MUST be lazily loaded on first `/api/generate` (or stream) request and cached thereafter, behind an `asyncio.Lock` to serialize concurrent calls.
- **REQ-010**: Model loading MUST NOT block server startup. `/api/status` MUST report `loading` during first load and `ready` thereafter so the SPA can poll and reflect state.
- **REQ-011**: The web server MUST bind to `127.0.0.1` only; it MUST NOT bind to `0.0.0.0` or any network interface reachable from the LAN. This is defense-in-depth even though the app is local-only.
- **REQ-012**: The web front-end MUST be a vanilla HTML/CSS/JS SPA with no JavaScript build step, no npm/node dependency, and no JS bundler. Total uncompressed client payload MUST be < 100 KB.
- **REQ-013`: The default server port MUST be `8000`. It MUST be overridable via `--port` CLI flag to `webapp.py`.
- **REQ-014**: WebSocket streaming MUST emit `meta` (sample_rate/channels) as the first server→client frame, followed by binary float32 PCM frames (little-endian, mono), followed by a `saved` JSON message with the URL of the assembled WAV file, followed by `done` with elapsed time. Errors MUST be signaled by an `error` JSON message.
- **REQ-015`: The `/api/generate` final-WAV response JSON MUST include: `file` (base filename), `url` (relative URL to fetch the WAV), `sample_rate` (int), `duration_s` (float), `elapsed_s` (float).

- **SEC-001**: Server MUST bind `127.0.0.1` only (per REQ-011). No CORS headers (`*`) MUST be configured.
- **SEC-002**: Uploaded reference files MUST be validated: only extensions `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg` are permitted; file size limit 25 MB per upload; files saved with a server-generated sanitized name (uuid + original extension).
- **SEC-003`: Static file serving of `output/` MUST only serve files with `.wav` extension and MUST reject path traversal (`..`, absolute paths) with HTTP 404.

- **CON-001**: Target runtime is Windows (PowerShell 7+ host). All file operations MUST use `pathlib.Path`. No platform-specific shell invocations in new code.
- **CON-002`: Project uses Python 3.12 in `.venv`. New code MUST be compatible with Python 3.10 through 3.12 (no 3.13-only syntax).
- **CON-003**: The installed `VoxCPM` package (`C:\Users\xbox\dev\scratch\voxcpm-test\.venv\Lib\site-packages\voxcpm\`) exposes no `web/` module (verified); only `cli.py`, `core.py`, `__init__.py`, `model/`, `modules/`, `utils/`, `zipenhancer.py`. The web UI MUST be implemented entirely in this repo, NOT in the voxcpm package.
- **CON-004**: No new test framework will be added (project currently has none). Verification is manual + smoke (curl/Invoke-WebRequest) per Section 6.
- **CON-005**: Existing `output/` directory (containing 12 historical WAVs) MUST NOT be deleted; new generations continue to be timestamped via existing `make_unique_output_path()`.

- **GUD-001`: Follow existing code style in `app.py`: 4-space indent, double-quote string defaults, type hints on all public signatures, module-level constants in UPPER_SNAKE_CASE, `from __future__ import annotations` not required (3.12 target allows `|` union syntax already used in `app.py:49`).
- **GUD-002**: All new Python files MUST start with a triple-quoted module docstring describing purpose, mirroring `app.py:1-22` style.
- **GUD-003**: Avoid adding comments to new code (matching repo convention; `app.py` uses docstrings only, no inline `#` comments except as section dividers like `app.py:37-39`).
- **GUD-004**: Numeric/behavioral constants (default port, upload size limit, etc.) MUST be module-level UPPER_SNAKE constants, mirroring `app.py:41-46`.
- **GUD-005**: HTTP status codes MUST be explicit in responses (e.g., `JSONResponse(status_code=400, content={...})`); no implicit 200 from raise.

- **PAT-001**: Mirror the existing `load_model()` device-detection and CUDA-diagnostic logic in `app.py:179-215`. Do not reimplement device detection; import from `core.py`.
- **PAT-002`: Mirror the existing `generate_with_retry()` wrapper in `app.py:246-261` for `/api/generate`. For `/api/generate/stream`, retry is NOT applied (matches `app.py:441` `retry_badcase=False`).
- **PAT-003`: Mirror `build_generation_kwargs()` in `app.py:264-290` for constructing VoxCPM kwargs, including `reference_wav_path` only when provided.
- **PAT-004`: Mirror `save_wav()` in `app.py:218-224` and `make_unique_output_path()` in `app.py:102-120` exactly (preserves output filename pattern used by existing `output/` files like `voice_design_1_20260526_110103.wav`).
- **PAT-005`: Use `logging` module (not `print`) inside `webapp.py` for server-side diagnostic output, since uvicorn captures stdout; `app.py` may continue using `print`.

## 2. Implementation Steps

### Implementation Phase 1: Core module extraction

- GOAL-001: Create `core.py` containing all reusable TTS logic currently inline in `app.py`, exposing functions importable by both the CLI and the web server. `app.py` MUST be refactored to import from `core.py` with ZERO behavior change to existing CLI flags, defaults, or console output.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create new file `core.py` at repo root (`C:\Users\xbox\dev\scratch\voxcpm-test\core.py`). Start with module docstring: `"""Shared core for VoxCPM2 TTS: model loading, generation, persistence. Used by app.py CLI and webapp.py FastAPI server."""`. | | |
| TASK-002 | MOVE (literal cut/paste, do not modify bodies) the following symbols from `app.py` to `core.py`, preserving exact signatures and behavior: `OUTPUT_DIR` (line 41), `DEFAULT_CFG` (44), `DEFAULT_STEPS` (45), `STREAMING_STEPS` (46), `detect_nvidia_gpus()` (49-64), `print_cuda_diagnosis()` (67-95), `ensure_output_dir()` (98-99), `make_unique_output_path()` (102-120), `detect_device()` (123-137), `get_runtime_device()` (140-159), `ensure_model_device()` (162-176), `load_model()` (179-215), `save_wav()` (218-224), `is_bad_waveform()` (227-243), `generate_with_retry()` (246-261), `build_generation_kwargs()` (264-290). | | |
| TASK-003 | ADD to `core.py` a new constant: `UPLOAD_DIR = Path("uploads")`. | | |
| TASK-004 | ADD to `core.py` a new function `ensure_upload_dir() -> None` with body `UPLOAD_DIR.mkdir(exist_ok=True)`, mirroring `ensure_output_dir()` pattern from `app.py:98-99`. | | |
| TASK-005 | ADD to `core.py` a new function `list_outputs() -> list[dict]`. Body: call `ensure_output_dir()`, then for each `*.wav` file in `OUTPUT_DIR.iterdir()`, return dict `{"name": f.name, "size_bytes": f.stat().st_size, "mtime_iso": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}`. Sort by `mtime_iso` descending. Import `datetime` at module top (move from existing `from datetime import datetime` already present). | | |
| TASK-006 | ADD to `core.py` a new function `list_uploads() -> list[dict]` with identical shape and behavior as `list_outputs()` but iterating `UPLOAD_DIR`. Call `ensure_upload_dir()` first. | | |
| TASK-007 | ADD to `core.py` a new function `save_upload(original_filename: str, data: bytes) -> Path`. Body: (a) call `ensure_upload_dir()`; (b) validate extension of `original_filename` is in `ALLOWED_UPLOAD_EXTS` (new constant: `frozenset({".wav",".flac",".mp3",".m4a",".ogg"})`); raise `ValueError("Unsupported file type")` otherwise; (c) generate sanitized name `f"{uuid.uuid4().hex}{suffix}"` (import `uuid` at top); (d) write `data` to `(UPLOAD_DIR / sanitized_name).write_bytes(data)`; (e) return the path. | | |
| TASK-008 | ADD to `core.py` a new class `ModelState` with attributes: `model: Any = None`, `state: str = "uninitialized"` (one of `"uninitialized"`, `"loading"`, `"ready"`, `"error"`), `lock: asyncio.Lock` (lazy-initialized, see TASK-009), `load_denoiser: bool = False`. Methods: `async def get_or_load(self, load_denoiser: bool = False) -> Any` that locks, sets state `"loading"`, calls `load_model()` from this module inside `await asyncio.to_thread(...)`, sets `self.model`, sets state `"ready"`, returns `self.model`. On exception, set state `"error"` and re-raise. Import `asyncio` at top. | | |
| TASK-009 | ADD to `core.py` a new function `streaming_adapter(model, params: dict) -> AsyncIterator` (an async generator) that runs the blocking `model.generate_streaming(**params)` generator inside `asyncio.to_thread` chunk-by-chunk, yielding each `np.ndarray` chunk. Implementation signature: `async def streaming_adapter(model, params: dict):` containing an `asyncio.Queue` and a background task that pulls from the sync generator via `await asyncio.to_thread(next, iterator)` until `StopIteration`. Use a sentinel object `class _Sentinel: pass` for completion signaling. | | |
| TASK-010 | EDIT `app.py`: replace the cut function bodies with imports. Add after existing imports (around line 31): `from core import (DEFAULT_CFG, DEFAULT_STEPS, STREAMING_STEPS, OUTPUT_DIR, detect_nvidia_gpus, print_cuda_diagnosis, ensure_output_dir, make_unique_output_path, detect_device, get_runtime_device, ensure_model_device, load_model, save_wav, is_bad_waveform, generate_with_retry, build_generation_kwargs)`. Do NOT change `demo_voice_design`, `demo_voice_clone`, `demo_streaming`, `show_info`, `parse_args`, `main` function bodies except where they reference a moved symbol — those references now resolve via the import and work unchanged. | | |
| TASK-011 | REGRESSION CHECK: run `python app.py --info` and verify output matches pre-refactor format (Python version, PyTorch version, CUDA availability, MPS, candidate device, VoxCPM version). Run `python -c "import core; print('core imports clean')"` and `python -c "import app; print('app imports clean')"`. Both MUST exit 0. | | |

### Implementation Phase 2: FastAPI server skeleton

- GOAL-002: Create `webapp.py` with uvicorn entrypoint, lifespan managing a singleton `ModelState`, JSON status/info routes, and static mounts for `web/` and `output/`. After this phase, the server starts and `/api/status` returns `{"state": "uninitialized"}`.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-012 | Create new file `webapp.py` at repo root. Module docstring: `"""FastAPI web server for VoxCPM2 TTS. Local-only (binds 127.0.0.1). Run with: python webapp.py [--port N]"""`. | | |
| TASK-013 | TOP-OF-FILE constants in `webapp.py`: `DEFAULT_PORT = 8000`, `HOST = "127.0.0.1"`, `MAX_UPLOAD_BYTES = 25 * 1024 * 1024`. | | |
| TASK-014 | Imports in `webapp.py`: `from contextlib import asynccontextmanager`, `import argparse`, `import asyncio`, `import logging`, `from pathlib import Path`, `from fastapi import FastAPI, HTTPException, UploadFile, File, Request, WebSocket, WebSocketDisconnect`, `from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse`, `from fastapi.staticfiles import StaticFiles`, `import numpy as np`, `from core import (ModelState, OUTPUT_DIR, UPLOAD_DIR, list_outputs, list_uploads, save_upload, save_wav, build_generation_kwargs, generate_with_retry, show_info, detect_device, get_runtime_device, streaming_adapter)` (note: `show_info` may need refactor to return dict rather than print — see TASK-015). | | |
| TASK-015 | ADD to `core.py` a new function `get_runtime_info() -> dict` returning JSON-serializable dict mirroring `show_info()` output in `app.py:455-476`: keys `python`, `pytorch`, `cuda_available`, `cuda_version`, `cuda_gpus` (list of names), `mps_available`, `device`, `voxcpm_version`. Implementation: copy `show_info()` body, build dict instead of printing; keep `show_info()` as-is (CLI still uses it). | | |
| TASK-016 | In `webapp.py`, define `logger = logging.getLogger("voxcpm.webapp")` and `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")`. Define `model_state = ModelState()`. | | |
| TASK-017 | In `webapp.py`, define `@asynccontextmanager async def lifespan(app: FastAPI):` that logs "Server starting" on entry, logs "Server shutting down" on exit; sets `app.state.model_state = model_state`. Pass `lifespan=lifespan` to `FastAPI(...)`. | | |
| TASK-018 | In `webapp.py`, create `app = FastAPI(title="VoxCPM2 TTS", version="1.0", lifespan=lifespan)`. | | |
| TASK-019 | ADD `@app.get("/api/status")` route returning `JSONResponse({"state": model_state.state})`. No request body. | | |
| TASK-020 | ADD `@app.get("/api/info")` route returning `JSONResponse(get_runtime_info())`. Implementation: call `await asyncio.to_thread(get_runtime_info)` (CPU-bound as it imports torch). | | |
| TASK-021 | ADD static mounts AFTER all API routes (order matters: API routes take precedence). `app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")` — wrap in `try/except` calling `ensure_output_dir()` first via `from core import ensure_output_dir`. Similarly for `web/` directory: `WEB_DIR = Path(__file__).parent / "web"` (will exist after Phase 6 — add to `webapp.py` now with a `try/except` to skip mount if missing, so the file imports during Phase 2 smoke tests). | | |
| TASK-022 | ADD `if __name__ == "__main__":` block at bottom of `webapp.py`. Parse `--port` (int, default `DEFAULT_PORT`) and optional `--reload` flag. Call `import uvicorn; uvicorn.run("webapp:app", host=HOST, port=args.port, reload=args.reload)`. Note: when `reload=True`, pass the import string; otherwise `uvicorn.run(app, ...)` is fine. Use the former for simplicity. | | |
| TASK-023 | SMOKE TEST: `python webapp.py --port 8000` (run in background; kill after). `curl http://127.0.0.1:8000/api/status` MUST return `{"state":"uninitialized"}`. `curl http://127.0.0.1:8000/api/info` MUST return a JSON object containing keys `python`, `pytorch`, `cuda_available`. Server log MUST show "Uvicorn running on http://127.0.0.1:8000". | | |

### Implementation Phase 3: Final-WAV generation endpoint and uploads

- GOAL-003: Implement `POST /api/generate` (synchronous final WAV) and `POST /api/uploads` / `GET /api/uploads` for managing reference audio. Verify by curl with JSON body.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-024 | ADD to `core.py` a pydantic `BaseModel` reflecting REQ-005 parameter subset. Class name: `GenerationParams`. Fields (all with defaults where indicated): `text: str`, `voice_description: str | None = None`, `reference_wav_path: str | None = None`, `cfg_value: float = 2.0`, `inference_timesteps: int = 20`, `normalize: bool = True`, `attempts: int = 2`, `seed: int | None = None`. Import `from pydantic import BaseModel`. Add validators: `0.5 <= cfg_value <= 4.0`, `1 <= inference_timesteps <= 50`, `1 <= attempts <= 10`. Use `pydantic.Field(..., ge=, le=)` constraints. | | |
| TASK-025 | ADD to `core.py` a function `assemble_prompt(params: GenerationParams) -> str` that returns: (a) if `params.voice_description` is not None: `f"({params.voice_description}){params.text}"`; (b) else `params.text`. This mirrors the prompt construction in `app.py:333`. | | |
| TASK-026 | ADD to `core.py` a function `apply_seed(seed: int | None) -> None` matching `app.py:591-597`: if seed is None, return; else `torch.manual_seed(seed)` and `if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)`. Import `torch` lazily inside function (matches `app.py` pattern). | | |
| TASK-027 | ADD to `core.py` an async function `generate_final(model_state: ModelState, params: GenerationParams) -> dict`. Body: (a) `apply_seed(params.seed)`; (b) `model = await model_state.get_or_load()`; (c) `prompt = assemble_prompt(params)`; (d) build kwargs via `build_generation_kwargs(text=prompt, cfg_value=params.cfg_value, inference_timesteps=params.inference_timesteps, normalize=params.normalize, min_len=2, max_len=4096, retry_badcase=True, retry_badcase_max_times=5, retry_badcase_ratio_threshold=5.0, reference_wav_path=params.reference_wav_path)`; (e) acquire `model_state.lock` and run `wav = await asyncio.to_thread(generate_with_retry, model, attempts=params.attempts, **kwargs)`; (f) `path = await asyncio.to_thread(save_wav, wav, "ui_generate.wav", model.tts_model.sample_rate)`; (g) return `{"file": path.name, "url": f"/outputs/{path.name}", "sample_rate": model.tts_model.sample_rate, "duration_s": len(wav)/model.tts_model.sample_rate, "elapsed_s": elapsed}` where `elapsed` is computed by recording `t0 = time.perf_counter()` at step (c) start and `elapsed = time.perf_counter() - t0` at (g). Import `time` at top. | | |
| TASK-028 | In `webapp.py`, ADD `@app.post("/api/generate")` route. Body: `async def generate(params: GenerationParams):` that wraps `await generate_final(model_state, params)` in try/except. On `Exception as e`: log error, return `JSONResponse(status_code=500, content={"error": str(e)})`. On success, return `JSONResponse(result)`. | | |
| TASK-029 | In `webapp.py`, ADD `@app.post("/api/uploads")` route. Body: `async def upload_reference(reference: UploadFile = File(...)):` validate `len(await reference.read()) <= MAX_UPLOAD_BYTES` (read into bytes once, check size, then pass bytes to `save_upload`). Validate extension against `ALLOWED_UPLOAD_EXTS` via `Path(reference.filename).suffix.lower()`. On validation failure, return `JSONResponse(status_code=400, content={"error": "..."})`. On success: `path = await asyncio.to_thread(save_upload, reference.filename, data)` and return `JSONResponse({"path": str(path), "name": path.name})`. | | |
| TASK-030 | In `webapp.py`, ADD `@app.get("/api/uploads")` route. Body: returns `JSONResponse(await asyncio.to_thread(list_uploads))`. | | |
| TASK-031 | SMOKE TEST (no model load required for status check; generation test requires model warmup): `curl http://127.0.0.1:8000/api/uploads` MUST return `[]`. Upload a small WAV: `curl -F "reference=@output/disco-shoulders-lady.wav" http://127.0.0.1:8000/api/uploads` MUST return JSON with `path`/`name`. Subsequent `curl http://127.0.0.1:8000/api/uploads` MUST return a one-element array. After model warmup, `curl -X POST http://127.0.0.1:8000/api/generate -H "Content-Type: application/json" -d '{"text":"Hello from web UI","cfg_value":2.0,"inference_timesteps":20}'` MUST return a JSON object with `url`/`sample_rate`/`duration_s`/`elapsed_s`. Fetch the returned `url` and confirm it serves a WAV. | | |

### Implementation Phase 4: Output history endpoints

- GOAL-004: Expose `/api/outputs` listing persisted WAV files for the SPA history panel. The static mount from TASK-021 already serves individual files.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-032 | In `webapp.py`, ADD `@app.get("/api/outputs")` returning `JSONResponse(await asyncio.to_thread(list_outputs))`. | | |
| TASK-033 | (Optional, hardening) ADD `@app.get("/api/outputs/{name}")` route handler that validates `name` is a single filename with `.wav` extension and no path separators (regex `^[A-Za-z0-9_.-]+\.wav$`), else `JSONResponse(status_code=404, content={"error":"not found"})`. On match, return `FileResponse(OUTPUT_DIR / name)`. If this exists, the static mount from TASK-021 is still preferable for general serving; this route provides explicit 404 behavior for malformed input. | | |
| TASK-034 | SMOKE TEST: `curl http://127.0.0.1:8000/api/outputs` MUST return JSON array of 12+ entries (existing `output/` files plus any new generations). Each entry MUST have keys `name`, `size_bytes`, `mtime_iso`. `curl http://127.0.0.1:8000/outputs/voice_design_1-fail.wav` MUST serve audio (HTTP 200, `Content-Type: audio/wav`). `curl http://127.0.0.1:8000/outputs/../../etc/passwd` MUST return HTTP 404 or 400 (StaticFiles already protects, but verify). | | |

### Implementation Phase 5: WebSocket streaming endpoint (server)

- GOAL-005: Implement `/api/generate/stream` WebSocket route that consumes a `start` JSON message with `GenerationParams`, emits `meta`, then binary float32 PCM chunks, then `saved` URL, then `done`. Verify via a small Python client script.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-035 | In `webapp.py`, ADD `@app.websocket("/api/generate/stream")` route. Body: `async def generate_stream(ws: WebSocket):` that (a) calls `await ws.accept()`; (b) `msg = await ws.receive_json()`; (c) validates `msg.get("type") == "start"` (else send `{type:"error",message:"expected start"}` + close); (d) `params = GenerationParams(**msg["params"])` (on validation error, send `{type:"error",message:str(e)}` + close); (e) `apply_seed(params.seed)`; (f) `model = await model_state.get_or_load()`; (g) send `{type:"meta", sample_rate: model.tts_model.sample_rate, channels: 1}` to client; (h) `prompt = assemble_prompt(params)`; (i) build kwargs (same as TASK-027 step (d) but with `retry_badcase=False` and `inference_timesteps` defaulting to STREAMING_STEPS=12 if params.inference_timesteps is the default 20 — see TASK-036). | | |
| TASK-036 | TASK-036 fine-tune: streamingMode uses `inference_timesteps` from client if provided else `STREAMING_STEPS` (12). Document this in the SPA UI by using `12` as the default placeholder on the streaming-specific input field (Phase 6 / TASK-043). | | |
| TASK-037 | In TASK-035 step (j), SET `chunks: list[np.ndarray] = []`; iterate `async for chunk in streaming_adapter(model, kwargs):` (TASK-009) — for each chunk: (1) `chunks.append(chunk)` (2) `bytes_payload = chunk.astype(np.float32).tobytes()` (3) `await ws.send_bytes(bytes_payload)` (4) optionally `await ws.send_json({type:"progress", chunk_samples: len(chunk), total_samples: sum(len(c) for c in chunks)})` after each chunk. On generator completion: assemble `wav = np.concatenate(chunks)`, `path = await asyncio.to_thread(save_wav, wav, "ui_stream.wav", model.tts_model.sample_rate)`, send `{type:"saved", url: f"/outputs/{path.name}", total_duration_s: len(wav)/model.tts_model.sample_rate}`, then `{type:"done", elapsed_s: <computed>}`. | | |
| TASK-038 | WRAP TASK-035 body in try/except `WebSocketDisconnect` (just `return`), plus catch-all `Exception as e`: log, `await ws.send_json({type:"error", message: str(e)})`, `await ws.close()`. Ensure the `asyncio.Lock` in `model_state` is released on any error (use `async with model_state.lock:` around step (j) — BUT note `model_state.get_or_load()` already acquires the lock; refactor so get_or_load releases its lock after model is loaded, then reacquire for streaming). Clarification: `model_state.lock` MUST be held for the DURATION of generation (final or streaming), NOT just for loading. Refactor TASK-009 load path: `get_or_load()` uses its own internal lock OR the same lock but releases after loading; the caller wraps actual generation in `async with model_state.lock:`. Implement as: `model_state.get_or_load(load_denoiser=False)` uses an internal `_load_lock` (separate), and `model_state.lock` (the public one) is for serializing generation calls. Update TASK-008 accordingly. | | |
| TASK-039 | SMOKE TEST: write `webtest_stream_client.py` (throwaway) that uses `websockets` lib (already pulled in via `uvicorn[standard]`), connects to `ws://127.0.0.1:8000/api/generate/stream`, sends `{type:"start", params:{text:"Streaming test.", inference_timesteps:12}}`, verifies reception of `meta` frame, verifies at least one binary frame, verifies `saved` frame with valid URL, fetches the URL and confirms WAV is playable. Delete `webtest_stream_client.py` after verification. | | |

### Implementation Phase 6: SPA non-streaming UI

- GOAL-006: Build the vanilla HTML/CSS/JS SPA. After this phase, the browser at `http://127.0.0.1:8000/` supports final-WAV generate, uploads, history list, and playback.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-040 | Create directory `web/` in repo root. | | |
| TASK-041 | Create `web/index.html` with: (a) `<!DOCTYPE html>`, `<html lang="en">`, `<head>` with `<meta charset="utf-8">`, `<meta name="viewport" content="width=device-width, initial-scale=1">`, `<title>VoxCPM2 TTS</title>`, `<link rel="stylesheet" href="/style.css">`; (b) `<body>` containing: a status banner `<div id="status">Connecting…</div>`, a `<form id="generate-form">` with fields matching `GenerationParams` (textarea#text, input#voice_description, input[type="file"]#reference_upload, range/slider for cfg_value, number for inference_timesteps, checkbox for normalize, number for attempts, number for seed), a `<button type="submit" id="generate-btn">Generate</button>`, a `<button type="button" id="stream-btn">Generate & Stream Live</button>` (disabled in this phase — wired in TASK-048), `<audio id="player" controls>` for playback, and `<ul id="history">` for past outputs. At end of `<body>`: `<script src="/app.js"></script>`. Form layout uses semantic fieldset/legend. | | |
| TASK-042 | Create `web/style.css` with minimal styling: system font stack (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`), max width 800 px container centered, padded cards for form and history, monospace font for status, indeterminate "Loading…"-style spinner CSS for `#status[state="loading"]`, readonly state for `#status[state="ready"]` (green text), `[state="error"]` red. File MUST be < 5 KB. | | |
| TASK-043 | Create `web/app.js` with vanilla JS (no modules, no imports). Top-level IIFE: `(function(){ 'use strict'; ... })();`. Define helpers: `async function getJSON(url)` returning parsed JSON or throwing. `async function postJSON(url, body)` setting `Content-Type: application/json`. `function setStatus(state, text)` updating `#status` text and `state` attribute. | | |
| TASK-044 | In `web/app.js`, on `DOMContentLoaded`: (a) call `getJSON('/api/info')` and render device info into the status banner; (b) initiate `pollStatus()` loop that polls `getJSON('/api/status')` every 1000 ms; when `state==="ready"` or `"uninitialized"`, set banner accordingly; when `"loading"`, show spinner; on first `"ready"`, stop polling; (c) call `loadHistory()` and `loadUploads()`. | | |
| TASK-045 | In `web/app.js`, implement `onGenerateFormSubmit(event)`: `event.preventDefault()`; collect form values into a `GenerationParams`-shaped object (`text`, `voice_description` or null, `reference_wav_path` null in this phase since download via UI form is TASK-046, `cfg_value` as float, `inference_timesteps` as int, `normalize` bool, `attempts` int, `seed` int or null); `postJSON('/api/generate', params)`; on success, set `#player.src = result.url` and call `loadHistory()`; on error, display error text in status banner. Disable the submit button while in-flight. | | |
| TASK-046 | Add upload handling in `web/app.js`: intercept the `#reference_upload` file input `change` event; `POST` the file via `FormData` to `/api/uploads`; on success, store the returned `name`/`path` in a module-scoped variable `selectedReferencePath` and display a `<div>` showing "Using reference: <name>" with a "remove" button to clear it. Wire `selectedReferencePath` into `onGenerateFormSubmit` so the request body includes `reference_wav_path: selectedReferencePath.name`. | | |
| TASK-047 | In `web/app.js`, `loadHistory()` calls `getJSON('/api/outputs')`, sorts by `mtime_iso` descending (server already sorts but re-sort client-side defensively), renders `<li>` items containing `<audio src="/outputs/<name>" controls>` plus file name and human-readable size + mtime. `loadUploads()` similar but lists from `/api/uploads`. | | |
| TASK-048 | Leave `#stream-btn` with `disabled` attribute set; will be enabled in Phase 7 (TASK-049). Add a comment in HTML: `<!-- Stream button enabled in Phase 7 -->` (temporary, removed in TASK-049). | | |
| TASK-049 | (Placeholder for Phase 7; no action here.) | | |
| TASK-050 | SMOKE TEST: `python webapp.py --port 8000`, open browser to `http://127.0.0.1:8000/`. Verify: status banner shows "Ready" with device info, form renders, fill text field "Hello world" + default params, click Generate → button disables, on success audio player loads the new WAV and plays, history list refreshes with the new entry at top. Upload a reference WAV → it appears in uploads list, selecting it as reference and clicking Generate produces a clone-mode WAV. | | |

### Implementation Phase 7: WebSocket client (live streaming playback)

- GOAL-007: Enable the `#stream-btn`. On click, open a WebSocket to `/api/generate/stream`, play incoming float32 chunks via `AudioContext` for low-latency gapless playback, and save the final WAV URL into history.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-051 | In `web/app.js`, define `class StreamPlayer` with: `constructor()` creating `this.audioCtx = null`, `this.nextStartTime = 0`, `this.receivedChunks = 0`. Method `ensureContext(sampleRate)`: if `this.audioCtx` is null, `this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: sampleRate})` (NOTE: setting sample rate requires AudioContext constructor support; if it throws, fall back to default and rely on resampling). Method `enqueueChunk(float32Array)`: create `AudioBufferSourceNode`, `buffer = audioCtx.createBuffer(1, float32Array.length, audioCtx.sampleRate)`, `buffer.getChannelData(0).set(float32Array)`, `source.buffer = buffer`, `source.connect(audioCtx.destination)`, schedule `source.start(this.nextStartTime)`, update `this.nextStartTime = Math.max(this.nextStartTime, audioCtx.currentTime) + buffer.duration`. Method `close()`: stop all scheduled sources after `nextStartTime`. | | |
| TASK-052 | In `web/app.js`, implement `onStreamBtnClick(event)`: (a) collect same form params as onGenerateFormSubmit; (b) construct WebSocket `new WebSocket('ws://' + location.host + '/api/generate/stream')`; (c) on `open`, send `{type:"start", params: {...}}` as JSON string; (d) on `message`, switch on `event.data`: if `event.data instanceof Blob`, `arrayBuffer = await event.data.arrayBuffer()`, `float32 = new Float32Array(arrayBuffer)` (byte order matches server-side little-endian float32), call `player.enqueueChunk(float32)`; if string, parse JSON, switch on `data.type`: `meta` → `player.ensureContext(data.sample_rate)` and resume `audioCtx` (Chrome requires user-gesture); `progress` → update a `<div id="stream-progress">` text; `saved` → push `data.url` to a `finalUrl` var; `done` → call `loadHistory()`, set `#player.src = finalUrl`, re-enable stream button; `error` → alert + re-enable. (e) on `error` or `close`, re-enable buttons. | | |
| TASK-053 | REMOVE the `disabled` attribute from `#stream-btn` and remove the placeholder note from TASK-048. Wire `#stream-btn` `click` → `onStreamBtnClick`. | | |
| TASK-054 | SMOKE TEST: open the browser UI; click "Generate & Stream Live" with text "Streaming synthesis test." Verify: (a) playback begins BEFORE the console shows `done` (proves streaming is live); (b) on completion, `<audio>` element contains the persisted WAV at `/outputs/ui_stream_<timestamp>.wav`; (c) history list includes the new streaming output. | | |

### Implementation Phase 8: Documentation, packaging, gitignore

- GOAL-008: Update `README.md`, `requirements.txt`, and `.gitignore` to reflect the new web UI. No new test framework or build infrastructure.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-055 | EDIT `requirements.txt`: replace contents with `voxcpm`, `soundfile`, `numpy`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic` (one per line, no version pins — match existing style). Note: these are all already installed in `.venv` (verified); pinning is deferred until a future packaging phase. | | |
| TASK-056 | EDIT `.gitignore`: under the "Generated outputs" section, add a new section "# Web uploads / reference audio\nuploads/" after line 27 (`output/`). | | |
| TASK-057 | EDIT `README.md`: after the existing "## Notes" section (end of file), APPEND the new section: `## Web UI\n\nLocal-only FastAPI server for interactive generation.\n\n\`\`\`bash\n.venv/bin/python webapp.py --port 8000\n# Open http://127.0.0.1:8000\n\`\`\`\n\nSupports voice design, voice cloning (upload reference), and live streaming playback.\n\n### API endpoints\n\n- GET  /api/status\n- GET  /api/info\n- POST /api/generate\n- WS   /api/generate/stream\n- GET  /api/outputs\n- GET  /api/uploads\n- POST /api/uploads\n\n### Constraints\n\nSingle user, localhost-only (binds 127.0.0.1). Reference uploads saved to \`uploads/\` (gitignored).`. Use the same backtick style as existing README sections. | | |

### Implementation Phase 9: End-to-end verification

- GOAL-009: Verify all six top-level requirements end-to-end via manual browser and curl checks. Confirm no regression in CLI behavior.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-058 | CLI regression: run each of the commands documented in `README.md` lines 28-53 and confirm output matches pre-refactor: `python app.py --info`, `python app.py`, `python app.py --text "Welcome."`, `python app.py --clone --reference output/disco-shoulders-lady.wav --text "Clone test."`, `python app.py --stream --text "Stream test."`. Inspect `output/` for new timestamped WAVs. | | |
| TASK-059 | Web UI REQ-001 (voice design via UI): open browser, type free-form text + optional voice description "warm female voice", click Generate, verify playback + history. | | |
| TASK-060 | Web UI REQ-002 (voice clone via UI): upload `output/disco-shoulders-lady.wav` as reference, type clone text, click Generate, verify clone-mode WAV is produced (compare duration/character to verify). | | |
| TASK-061 | Web UI REQ-003 (live streaming): click "Generate & Stream Live", verify audio begins before `done` event fires (compare wall-clock: first audible playback time vs total synthesis time). | | |
| TASK-062 | Web UI REQ-004 (mode choice): both Generate (final WAV) and Generate & Stream Live buttons work and produce distinct output files. | | |
| TASK-063 | Web UI REQ-005 (param subset): vary `cfg_value` (1.5, 2.0, 2.5) and `inference_timesteps` (10, 20), verify outputs differ; verify `seed` produces repeatable outputs for two consecutive calls. | | |
| TASK-064 | Security REQ-011: confirm `netstat -an | findstr :8000` (Windows) shows `127.0.0.1:8000` LISTENING only, NOT `0.0.0.0:8000`. | | |
| TASK-065 | Security REQ-002: attempt uploading a file with extension `.txt` → expect HTTP 400 with `{"error":"Unsupported file type"}`. Attempt uploading a 30 MB WAV → expect HTTP 400 with size limit error. | | |
| TASK-066 | Security REQ-003: `curl http://127.0.0.1:8000/outputs/..%2Fapp.py` returns 404 (path traversal blocked). | | |
| TASK-067 | Final review: inspect `git status` — confirm new files are `core.py`, `webapp.py`, `web/index.html`, `web/app.js`, `web/style.css`, and modified files are `app.py`, `requirements.txt`, `.gitignore`, `README.md`. Confirm `uploads/` and `output/` are NOT staged (gitignored). | | |

## 3. Alternatives

- **ALT-001**: **Gradio web UI (rejected)**. VoxCPM2's own official web demo uses Gradio, and Gradio 6.12.0 is already installed in the venv. This would have been the lowest-effort path (~1 file, built-in audio/upload components, built-in WebSocket-like streaming via `gr.Audio(streaming=True)`). Rejected per user decision in favor of FastAPI + minimal SPA to retain full control over the client-side Web Audio scheduling and avoid Gradio's opinionated styling. Trade-off: more client-side code to maintain.
- **ALT-002**: **NiceGUI (rejected)**. A Python-only reactive UI on top of FastAPI — a middle ground between Gradio's batteries-included approach and a hand-written SPA. Rejected because it introduces a new dependency and still constrains client-side control; the user preferred a pure-FastAPI backend so the SPA stays framework-free.
- **ALT-003**: **Streamlit (rejected)**. A notebook-style alternative commonly used for ML demos. Rejected because (a) it's weaker for real-time chunked audio streaming, requiring workarounds for the streaming requirement (REQ-003), and (b) it introduces a new heavy dependency conflicting with the "vanilla SPA, no build step" constraint (REQ-012).
- **ALT-004**: **Shell out to `python app.py` from web server (rejected)**. The web server could spawn `app.py` as a subprocess per request and capture the produced WAV. This would be the simplest reuse path (zero refactor) but requires reloading the multigigabyte VoxCPM model on every request — unacceptable latency (5-20 s warmup per call) and VRAM churn. The chosen approach loads the model once per process (REQ-009).
- **ALT-005**: **Replace `app.py` CLI entirely with web UI (rejected)**. Cleaner repo (one entrypoint) but loses scriptability and CI hooks for the existing CLI flags documented in `README.md`. User explicitly chose to keep CLI alongside web UI (REQ-008).
- **ALT-006**: **JS framework SPA (React/Vue/Svelte) with build step (rejected)**. Greater long-term flexibility for complex UI (e.g., waveform visualization, multi-voice management), but adds npm/build complexity to a Python-only local tool and inflates deployment footprint. Rejected in favor of vanilla JS (REQ-012) for a UI with < 100 KB client payload.
- **ALT-007**: **Eager model load at server startup (rejected)**. Alternative to lazy load (REQ-010). Eager load makes the first generation request instant but delays server startup by 5-20 s, complicating development iteration and obscuring model-load failures. Lazy load with status polling was chosen for transparency and fast startup.

## 4. Dependencies

- **DEP-001**: `voxcpm` (PyPI, version 2.0.x) — already in venv and `requirements.txt`. Provides `VoxCPM.from_pretrained()`, `model.generate()`, `model.generate_streaming()`, `model.tts_model.sample_rate` (see `docs/VOXCPM2_RESEARCH.md` Section 4).
- **DEP-002**: `torch` (2.11.0+cu128 in venv) — provides device detection, `torch.manual_seed`, `torch.cuda` API. Already transitively installed via voxcpm.
- **DEP-003**: `torchaudio` (2.11.0+cu128 in venv) — used internally by voxcpm for reference audio loading. Already transitively installed.
- **DEP-004**: `numpy` (2.4.4 in venv) — waveform arrays. Already in `requirements.txt`.
- **DEP-005**: `soundfile` (already in `requirements.txt`) — WAV I/O via `save_wav()` in `core.py`.
- **DEP-006**: `fastapi` (0.135.3 in venv) — NEW entry in `requirements.txt` (TASK-055). Required version: `>=0.100` (modern lifespan API).
- **DEP-007**: `uvicorn[standard]` (0.44.0 in venv) — NEW entry in `requirements.txt`. Provides `uvicorn.run()` plus `websockets` extra (transitively) for the `/api/generate/stream` route.
- **DEP-008**: `python-multipart` — NEW entry. Required by FastAPI for `UploadFile` / `File(...)` route in TASK-029 (multipart form parsing).
- **DEP-009**: `pydantic` (transitively via fastapi) — for `GenerationParams` model (TASK-024).
- **DEP-010**: Browser support for `AudioContext` with `sampleRate` construction parameter (TASK-051). Modern Chrome/Firefox/Edge support this; Safari < 14.1 may ignore the requested sample rate and internal resampling will apply. Acceptable degradation.
- **DEP-011`: Browser support for `WebSocket` (universal) and `Float32Array` (universal on browsers supporting ES2017+).

## 5. Files

- **FILE-001**: `core.py` — NEW. Shared TTS logic extracted from `app.py` plus new `UPLOAD_DIR`/`ensure_upload_dir`/`list_outputs`/`list_uploads`/`save_upload`/`ModelState`/`streaming_adapter`/`GenerationParams`/`assemble_prompt`/`apply_seed`/`generate_final`/`get_runtime_info`.
- **FILE-002**: `app.py` — MODIFIED. Bodies of moved helpers removed and replaced with `from core import (...)`. `demo_voice_design` (lines 298-357), `demo_voice_clone` (360-406), `demo_streaming` (409-452), `show_info` (455-476), `parse_args` (484-579), `main` (582-657) UNCHANGED. Net: file shrinks by ~400 LOC to ~250 LOC.
- **FILE-003**: `webapp.py` — NEW. FastAPI entrypoint (`app`), `lifespan` context manager, `/api/status`, `/api/info`, `/api/generate` (POST), `/api/generate/stream` (WS), `/api/outputs`, `/api/outputs/{name}`, `/api/uploads` (GET, POST), static mounts, `if __name__ == "__main__":` uvicorn runner.
- **FILE-004**: `web/index.html` — NEW. SPA shell with form, two action buttons, audio player, history list.
- **FILE-005**: `web/app.js` — NEW. IIFE-wrapped vanilla JS. Helpers + status polling + form submit + upload handling + history refresh + `StreamPlayer` class + WebSocket client.
- **FILE-006**: `web/style.css` — NEW. Minimal styling, < 5 KB.
- **FILE-007**: `requirements.txt` — MODIFIED. Append `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic` to existing `voxcpm`, `soundfile`, `numpy`.
- **FILE-008**: `.gitignore` — MODIFIED. Add `uploads/` under the "Generated outputs" section.
- **FILE-009**: `README.md` — MODIFIED. Append "## Web UI" section after "## Notes".
- **FILE-010**: `uploads/` — NEW directory (created at runtime by `ensure_upload_dir()`). Gitignored.
- **FILE-011**: `webtest_stream_client.py` — TEMPORARY throwaway script created in TASK-039 and DELETED after smoke verification. Not committed.

## 6. Testing

- **TEST-001 (smoke, TASK-011)**: `python app.py --info` and `python -c "import core, app"` both exit 0; `--info` output byte-for-byte matches pre-refactor (Python/PyTorch/CUDA/MPS/device/VoxCPM lines).
- **TEST-002 (smoke, TASK-023)**: `curl http://127.0.0.1:8000/api/status` returns `{"state":"uninitialized"}`; `curl http://127.0.0.1:8000/api/info` returns JSON with `python`/`pytorch`/`cuda_available` keys; uvicorn log shows "running on http://127.0.0.1:8000".
- **TEST-003 (functional, TASK-031)**: Upload via `curl -F "reference=@output/disco-shoulders-lady.wav" http://127.0.0.1:8000/api/uploads` returns `{"path":"...","name":"..."}`; subsequent list returns 1 entry. `POST /api/generate` with `{"text":"Hello"}` returns valid `url`/`duration_s`.
- **TEST-004 (functional, TASK-034)**: `GET /api/outputs` returns 12+ entries (existing + new). `GET /outputs/<existing>.wav` serves audio with `Content-Type: audio/wav`. `GET /outputs/../../etc/passwd` returns 4xx.
- **TEST-005 (functional, TASK-039)**: Endpoint-to-endpoint WebSocket test via throwaway Python client: receives `meta`, ≥1 binary frame, `saved`, `done`.
- **TEST-006 (manual UI, TASK-050)**: Browser at `http://127.0.0.1:8000` shows status Ready, form works, Generate produces playable WAV, history updates, uploads work end-to-end.
- **TEST-007 (manual UI, TASK-054)**: "Generate & Stream Live" button produces audio that begins playback before the `done` message arrives (compare wall-clock timestamps in console logs).
- **TEST-008 (regression, TASK-058)**: All five README CLI examples produce expected outputs in `output/`.
- **TEST-009 (security, TASK-064)**: `netstat -an | findstr :8000` shows only `127.0.0.1:8000` LISTENING.
- **TEST-010 (security, TASK-065)**: Upload of `.txt` returns HTTP 400; upload of 30 MB WAV returns HTTP 400.
- **TEST-011 (security, TASK-066)**: `GET /outputs/..%2Fapp.py` returns HTTP 404.
- **TEST-012 (functional, TASK-063)**: Varying `cfg_value` and `inference_timesteps` produces audibly different outputs; same `seed` value across two calls produces byte-identical WAVs (or near-identical — note this depends on VoxCPM determinism and CUDA backends; documented in `docs/VOXCPM2_RESEARCH.md` Section 8).
- **TEST-013 (code quality, all phases)**: No new inline `# comments` in Python files (per GUD-003); no remaining placeholder TODOs in code; `web/style.css` < 5 KB uncompressed (per TASK-042); total client payload < 100 KB (per REQ-012).

## 7. Risks & Assumptions

- **RISK-001**: `VoxCPM.generate_streaming()` is a synchronous generator. Running it inside `asyncio.to_thread` with a queue (TASK-009) is well-understood, but if the generator holds the GIL for long stretches, UI responsiveness may suffer on CPU-only devices. Mitigation: device is CUDA on this host (torch 2.11.0+cu128), so chunks release the GIL during CUDA ops; verify with TASK-054 timing. If problematic, fall back to running the entire stream generator in a thread and signaling via `asyncio.Queue`.
- **RISK-002**: VoxCPM's `generate_streaming()` may raise inside the iterator (e.g., badcase failures with `retry_badcase=False` — matching `app.py:441`). The `streaming_adapter` must propagate exceptions to the WS route, which sends `{type:"error", ...}`. The client-side in-progress audio MAY have played partial chunks before the error; acceptable for v1 (document in README).
- **RISK-003**: Setting `AudioContext({sampleRate: 48000})` (TASK-051) is supported on Chrome 55+, Firefox 25+, but Safari may ignore the requested rate and resample to its default (44.1 kHz). This causes mild quality loss on Safari only; documented in DEP-010. Host is Windows with Chrome/Edge/Firefox available, so primary target unaffected.
- **RISK-004**: The `asyncio.Lock` serialization (REQ-009) means a second generate request waits for the first to complete. Since this is local-only single-user (constraint), this is intentional and acceptable. If multiple browser tabs or rapid double-clicks occur, requests queue — UI should disable buttons during in-flight requests (TASK-045, TASK-052 already specify this).
- **RISK-005**: Pydantic v2 (transitively via FastAPI 0.135) uses different validator syntax than v1. TASK-024 spec uses `Field(ge=, le=)` which is valid in v2 — but the validating `BaseModel` must be imported from `pydantic` (not `pydantic.v1`). Implementation must confirm `pydantic>=2` is installed: `pip show pydantic` (already verified installed).
- **RISK-006**: `app.py:246-261` (`generate_with_retry`) accesses `model.tts_model.sample_rate` BEFORE generation. If `model.tts_model` is None for any VoxCPM version mismatch, this raises `AttributeError`. The existing CLI relies on this; the refactor preserves the assumption (PAT-002) without defense.
- **RISK-007**: Path traversal via static mount (TASK-021, TASK-033) is bounded by `StaticFiles` defaults (Starlette escapes `..` segments). However, the optional explicit route in TASK-033 must additionally regex-validate the filename; otherwise it could be vulnerable to `/api/outputs/..%2Fapp.py`. Mitigation specified in TASK-033 via regex `^[A-Za-z0-9_.-]+\.wav$`.
- **ASSUMPTION-001**: The installed VoxCPM2 version exposes `model.tts_model.sample_rate`. Verified in `app.py:248,354,406,445,452` (already relies on this attribute). Research doc confirms `48000`.
- **ASSUMPTION-002**: `model.generate()` and `model.generate_streaming()` accept the kwargs listed in `build_generation_kwargs()` (PAT-003). Verified by `app.py` usage working in current testing.
- **ASSUMPTION-003**: The web UI is for ONE user only; concurrency control beyond an `asyncio.Lock` for serializing generation is out of scope. If the constraint changes, a queued worker model is needed (out of scope for v1).
- **ASSUMPTION-004**: `WebSocket.send_bytes` on the server side accepts `bytes`; `ws.receive_bytes`/`event.data` instanceof `Blob` on the client side handles binary. Standard FastAPI/JavaScript behavior; no special protocol negotiation needed.
- **ASSUMPTION-005**: The host machine (Windows) has at least 8 GB free VRAM and a CUDA-capable GPU (torch 2.11.0+cu128 indicates this is the case). CPU/MPS fallback paths exist via `detect_device()` but UI performance may be poor on CPU-only operation; documented in `docs/VOXCPM2_RESEARCH.md` Section 6.
- **ASSUMPTION-006`: `python-multipart` is required for FastAPI `UploadFile` to function. Without it, importing the route raises `AssertionError: The python-multipart library must be installed`. Tasks TASK-029 and TASK-055 acknowledge this dependency.
- **ASSUMPTION-007**: No automated test framework will be added in v1 (per CON-004). All verification is manual or smoke (curl/Invoke-WebRequest) per Section 6. A future hardening phase could add `tests/test_core_smoke.py` with a `unittest.mock.MagicMock` for `VoxCPM`.

## 8. Related Specifications / Further Reading

- [VoxCPM2 Research Report](./VOXCPM2_RESEARCH.md) — complete API surface, parameters, hardware requirements, known limitations. Authoritative for VoxCPM behavior.
- [VoxCPM PyPI page](https://pypi.org/project/voxcpm/)
- [VoxCPM GitHub Repository](https://github.com/OpenBMB/VoxCPM)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Starlette WebSocket documentation](https://www.starlette.io/websockets/)
- [MDN: Web Audio API — AudioBufferSourceNode](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode)
- [MDN: WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Pydantic V2 documentation](https://docs.pydantic.dev/latest/)
- [Uvicorn settings reference](https://www.uvicorn.org/settings/)
