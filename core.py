"""Shared core for VoxCPM2 TTS: model loading, generation, persistence.
Used by app.py CLI and webapp.py FastAPI server.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from pydantic import BaseModel, Field


OUTPUT_DIR = Path("output")
DEFAULT_CFG = 2.0
DEFAULT_STEPS = 20
STREAMING_STEPS = 12
UPLOAD_DIR = Path("uploads")
ALLOWED_UPLOAD_EXTS = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg"})
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def detect_nvidia_gpus() -> list[str]:
    """Return NVIDIA GPU names from nvidia-smi when available."""
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def print_cuda_diagnosis() -> None:
    """Print actionable hints when CUDA is expected but unavailable."""
    import torch

    if torch.cuda.is_available():
        return

    nvidia = detect_nvidia_gpus()
    if not nvidia:
        return

    torch_version = torch.__version__
    print("\nCUDA diagnosis:")
    print(f"- NVIDIA GPU(s) detected: {', '.join(nvidia)}")
    print(f"- Installed torch build: {torch_version}")

    if "+cpu" in torch_version:
        print("- This is a CPU-only PyTorch build, so CUDA cannot be used.")
    else:
        print("- CUDA runtime is unavailable to this PyTorch install.")

    print(
        "- Fix: install a CUDA-enabled PyTorch wheel in this venv, then rerun --info."
    )
    print("- Example (Windows, pip):")
    print("    py -3 -m pip uninstall -y torch torchvision torchaudio")
    print(
        "    py -3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
    )


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def make_unique_output_path(filename: str) -> Path:
    """Create a unique output filename with timestamp + collision fallback."""
    ensure_output_dir()

    original = Path(filename)
    stem = original.stem
    suffix = original.suffix or ".wav"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    candidate = OUTPUT_DIR / f"{stem}_{timestamp}{suffix}"
    if not candidate.exists():
        return candidate

    for i in range(1, 1000):
        candidate = OUTPUT_DIR / f"{stem}_{timestamp}_{i:03d}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError("Could not allocate a unique output filename")


def detect_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # Validate CUDA with a tiny allocation so we fail fast on bad driver/runtime.
        try:
            _ = torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as exc:
            print(f"Warning: CUDA detected but unusable ({exc}); falling back")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_runtime_device(model) -> str:
    """Best-effort extraction of the model runtime device string."""
    runtime = getattr(getattr(model, "tts_model", None), "device", None)
    if runtime is not None:
        return str(runtime)

    tts_model = getattr(model, "tts_model", None)
    if tts_model is not None and hasattr(tts_model, "parameters"):
        try:
            return str(next(tts_model.parameters()).device)
        except Exception:
            pass

    if hasattr(model, "parameters"):
        try:
            return str(next(model.parameters()).device)
        except Exception:
            pass

    return "unknown"


def ensure_model_device(model, target_device: str) -> None:
    """Best-effort model placement for frameworks that expose .to(...)."""
    if target_device not in ("cuda", "mps", "cpu"):
        return

    for candidate in (model, getattr(model, "tts_model", None)):
        if candidate is None:
            continue
        to_method = getattr(candidate, "to", None)
        if callable(to_method):
            try:
                to_method(target_device)
            except Exception:
                # Some wrappers expose .to but don't support direct placement.
                pass


def load_model(load_denoiser: bool = False):
    """Load VoxCPM2 with sensible defaults for the current platform.

    Note: VoxCPM handles device placement internally.
    On Apple Silicon, VoxCPM2 automatically falls back to MPS when CUDA
    is unavailable. We only use device detection to decide whether
    torch.compile optimization is safe to enable.
    """
    from voxcpm import VoxCPM

    device = detect_device()
    # torch.compile doesn't work well on MPS/CPU — only enable on CUDA
    optimize = device.startswith("cuda")

    print(
        f"Loading VoxCPM2 (detected device={device}, optimize={optimize}, denoiser={load_denoiser})"
    )
    t0 = time.perf_counter()

    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        optimize=optimize,
        load_denoiser=load_denoiser,
    )

    effective_device = get_runtime_device(model)

    # If CUDA is available but the model remained on CPU, force placement.
    if device == "cuda" and "cuda" not in effective_device:
        print("Detected CUDA but model is not on CUDA; attempting to move model...")
        ensure_model_device(model, "cuda")
        effective_device = get_runtime_device(model)
        print(f"Runtime device after move attempt: {effective_device}")

    elapsed = time.perf_counter() - t0
    print(f"Model loaded in {elapsed:.1f}s (runtime device={effective_device})")
    return model


def save_wav(wav: np.ndarray, filename: str, sample_rate: int = 48000):
    """Save a numpy waveform to a WAV file in the output directory."""
    path = make_unique_output_path(filename)
    sf.write(str(path), wav, sample_rate)
    duration = len(wav) / sample_rate
    print(f"Saved: {path}  ({duration:.2f}s, {sample_rate}Hz)")
    return path


def is_bad_waveform(wav: np.ndarray, sample_rate: int) -> bool:
    """Basic sanity checks to catch obvious failed generations."""
    if wav is None or wav.size == 0:
        return True
    if not np.isfinite(wav).all():
        return True

    duration = wav.size / sample_rate
    if duration < 0.15:
        return True

    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(np.square(wav))))
    if peak < 1e-4 or rms < 1e-5:
        return True

    return False


def generate_with_retry(model, *, attempts: int = 2, **kwargs):
    """Retry generation when output is obviously broken."""
    sample_rate = model.tts_model.sample_rate
    last_wav = None

    for attempt in range(1, max(1, attempts) + 1):
        wav = model.generate(**kwargs)
        last_wav = wav
        if not is_bad_waveform(wav, sample_rate):
            if attempt > 1:
                print(f"    Recovered on retry attempt {attempt}")
            return wav
        if attempt < attempts:
            print(f"    Attempt {attempt} looked bad; retrying...")

    return last_wav


def build_generation_kwargs(
    *,
    text: str,
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    min_len: int,
    max_len: int,
    retry_badcase: bool,
    retry_badcase_max_times: int,
    retry_badcase_ratio_threshold: float,
    reference_wav_path: str | None = None,
):
    kwargs = dict(
        text=text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
        min_len=min_len,
        max_len=max_len,
        retry_badcase=retry_badcase,
        retry_badcase_max_times=retry_badcase_max_times,
        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
    )
    if reference_wav_path is not None:
        kwargs["reference_wav_path"] = reference_wav_path
    return kwargs


def show_info():
    """Print environment info for debugging."""
    import torch

    print("=== Environment Info ===")
    print(f"Python:     {sys.version}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.cuda.is_available()} ({torch.version.cuda or 'N/A'})")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            print(f"CUDA GPU {i}: {name}")
    print(f"MPS:        {torch.backends.mps.is_available()}")
    print(f"Device:     {detect_device()}")
    print_cuda_diagnosis()

    try:
        from importlib.metadata import version

        print(f"VoxCPM:     {version('voxcpm')}")
    except Exception:
        print("VoxCPM:     not installed")


def ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)


def list_outputs() -> list[dict]:
    ensure_output_dir()
    results = []
    for f in OUTPUT_DIR.iterdir():
        if f.suffix.lower() == ".wav":
            results.append({
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "mtime_iso": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    results.sort(key=lambda x: x["mtime_iso"], reverse=True)
    return results


def list_uploads() -> list[dict]:
    ensure_upload_dir()
    results = []
    for f in UPLOAD_DIR.iterdir():
        if f.suffix.lower() in ALLOWED_UPLOAD_EXTS:
            results.append({
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "mtime_iso": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    results.sort(key=lambda x: x["mtime_iso"], reverse=True)
    return results


def save_upload(original_filename: str, data: bytes) -> Path:
    ensure_upload_dir()
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise ValueError("Unsupported file type")
    sanitized = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / sanitized
    path.write_bytes(data)
    return path


def assemble_prompt(text: str, voice_description: str | None) -> str:
    if voice_description:
        return f"({voice_description}){text}"
    return text


def apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_runtime_info() -> dict:
    import torch
    from importlib.metadata import version as _version

    cuda_gpus = []
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            cuda_gpus.append(torch.cuda.get_device_name(i))

    voxcpm_version = "not installed"
    try:
        voxcpm_version = _version("voxcpm")
    except Exception:
        pass

    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "N/A",
        "cuda_gpus": cuda_gpus,
        "mps_available": torch.backends.mps.is_available(),
        "device": detect_device(),
        "voxcpm_version": voxcpm_version,
    }


class ModelState:
    def __init__(self):
        self.model: Any = None
        self.state: str = "uninitialized"
        self.lock: asyncio.Lock = asyncio.Lock()
        self._load_lock: asyncio.Lock = asyncio.Lock()

    def _init_locks(self):
        # Locks are now created eagerly in __init__ as instance attributes.
        # Kept for backward compatibility with existing call sites.
        pass

    async def get_or_load(self, load_denoiser: bool = False) -> Any:
        if self.model is not None:
            return self.model
        if self.state == "error":
            raise RuntimeError("Model failed to load")
        async with self._load_lock:
            if self.model is not None:
                return self.model
            self.state = "loading"
            try:
                self.model = await asyncio.to_thread(
                    load_model, load_denoiser=load_denoiser
                )
                self.state = "ready"
            except Exception:
                self.state = "error"
                raise
            return self.model


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


class _StreamError:
    """Wraps an exception from the background streaming thread for re-raising."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


async def streaming_adapter(model, params: dict):
    """Async iterator wrapping blocking model.generate_streaming().

    Exceptions raised by ``model.generate_streaming()`` are captured in the
    background thread and re-raised in the async generator so that consumers
    (e.g. the websocket handler) can surface them as errors instead of
    silently treating a failed stream as a successful empty stream.
    """
    queue: asyncio.Queue = asyncio.Queue()

    def _pull():
        try:
            for chunk in model.generate_streaming(**params):
                queue.put_nowait(chunk)
        except Exception as exc:
            queue.put_nowait(_StreamError(exc))
        finally:
            queue.put_nowait(_SENTINEL)

    asyncio.create_task(asyncio.to_thread(_pull))
    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, _StreamError):
            raise item.exc
        yield item


class GenerationParams(BaseModel):
    text: str
    voice_description: str | None = None
    reference_wav_path: str | None = None
    cfg_value: float = Field(default=2.0, ge=0.5, le=4.0)
    inference_timesteps: int = Field(default=20, ge=1, le=50)
    normalize: bool = True
    attempts: int = Field(default=2, ge=1, le=10)
    seed: int | None = None


async def generate_final(
    model_state: ModelState,
    params: GenerationParams,
    mode: str = "ui_generate",
) -> dict:
    apply_seed(params.seed)
    model = await model_state.get_or_load()

    t0 = time.perf_counter()
    prompt = assemble_prompt(params.text, params.voice_description)

    kwargs = build_generation_kwargs(
        text=prompt,
        cfg_value=params.cfg_value,
        inference_timesteps=params.inference_timesteps,
        normalize=params.normalize,
        min_len=2,
        max_len=4096,
        retry_badcase=True,
        retry_badcase_max_times=5,
        retry_badcase_ratio_threshold=5.0,
        reference_wav_path=params.reference_wav_path,
    )

    model_state._init_locks()
    async with model_state.lock:
        wav = await asyncio.to_thread(
            generate_with_retry, model, attempts=params.attempts, **kwargs
        )

    elapsed = time.perf_counter() - t0
    path = await asyncio.to_thread(save_wav, wav, f"{mode}.wav", model.tts_model.sample_rate)
    return {
        "file": path.name,
        "url": f"/outputs/{path.name}",
        "sample_rate": model.tts_model.sample_rate,
        "duration_s": len(wav) / model.tts_model.sample_rate,
        "elapsed_s": elapsed,
    }
