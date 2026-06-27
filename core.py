"""Shared core for VoxCPM2 TTS: model loading, generation, persistence.
Used by app.py CLI and webapp.py FastAPI server.
"""

import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from pydantic import BaseModel, Field


OUTPUT_DIR = Path("output")
JOBS_DIR = Path("jobs")
DEFAULT_CFG = 2.0
DEFAULT_STEPS = 20
STREAMING_STEPS = 12
UPLOAD_DIR = Path("uploads")
ALLOWED_UPLOAD_EXTS = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg"})
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

ENGINE_IDS = ("voxcpm", "supertonic", "neutts-nano", "kokoro")


def safe_slug(value: str | None, *, default: str = "job") -> str:
    raw = (value or default).strip().lower()
    cleaned = []
    last_dash = False
    for ch in raw:
        if ch.isalnum():
            cleaned.append(ch)
            last_dash = False
        elif ch in {"-", "_", " ", "."}:
            if not last_dash:
                cleaned.append("-")
                last_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or default


def timestamp_slug(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).strftime("%Y%m%d-%H%M%S")


def normalize_engine_id(engine_id: str) -> str:
    if engine_id == "neutts":
        return "neutts-nano"
    return engine_id


def module_is_installed(module_name: str) -> bool:
    """Check package presence without importing it and triggering side effects."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def prepare_kokoro_runtime() -> None:
    """Patch phonemizer compatibility expected by Kokoro/Misaki on some installs."""
    try:
        wrapper = importlib.import_module("phonemizer.backend.espeak.wrapper")
        espeak_wrapper = wrapper.EspeakWrapper
    except Exception:
        return
    if hasattr(espeak_wrapper, "set_data_path"):
        return

    @classmethod
    def set_data_path(cls, data_path):
        cls._ESPEAK_DATA_PATH = str(data_path)

    espeak_wrapper.set_data_path = set_data_path


def ensure_jobs_dir() -> None:
    JOBS_DIR.mkdir(exist_ok=True)


def _atomic_json_write(path: Path, data: dict) -> None:
    ensure_jobs_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:300] or exc.__class__.__name__


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


def normalize_wav_array(wav: np.ndarray) -> np.ndarray:
    """Normalize model audio output to a non-empty soundfile-compatible array."""
    arr = np.asarray(wav)
    if arr.size == 0:
        raise ValueError("Generated waveform is empty")
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.size == 0:
        raise ValueError("Generated waveform is empty")
    if not np.isfinite(arr).all():
        raise ValueError("Generated waveform contains non-finite values")
    return arr.astype(np.float32, copy=False)


def save_wav_exact_or_unique(wav: np.ndarray, filename: str, sample_rate: int = 48000):
    ensure_output_dir()
    wav = normalize_wav_array(wav)
    base = Path(filename)
    if base.suffix.lower() != ".wav":
        base = base.with_suffix(".wav")
    candidate = OUTPUT_DIR / base.name
    if candidate.exists():
        prefix = candidate.stem
        for i in range(1, 1000):
            alt = OUTPUT_DIR / f"{prefix}-{i:03d}.wav"
            if not alt.exists():
                candidate = alt
                break
    sf.write(str(candidate), wav, sample_rate)
    if candidate.stat().st_size <= 44:
        raise RuntimeError(f"Saved WAV is empty: {candidate}")
    return candidate


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
    wav = normalize_wav_array(wav)
    path = make_unique_output_path(filename)
    sf.write(str(path), wav, sample_rate)
    if path.stat().st_size <= 44:
        raise RuntimeError(f"Saved WAV is empty: {path}")
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


class EngineCapabilities(BaseModel):
    supports_plain_tts: bool = True
    supports_voice_design: bool = False
    supports_cloning: bool = False
    requires_reference_text: bool = False
    supports_preset_voices: bool = False
    supports_final_wav: bool = True
    supports_streaming: bool = False


class EngineStatus(BaseModel):
    state: str = "unloaded"
    load_error: str | None = None
    install_hint: str | None = None


class EngineInfo(BaseModel):
    engine_id: str
    display_name: str
    installed: bool
    status: EngineStatus
    capabilities: EngineCapabilities


class EngineRequest(BaseModel):
    text: str
    voice_description: str | None = None
    reference_wav_path: str | None = None
    reference_text: str | None = None
    cfg_value: float = Field(default=2.0, ge=0.5, le=4.0)
    inference_timesteps: int = Field(default=20, ge=1, le=50)
    normalize: bool = True
    attempts: int = Field(default=2, ge=1, le=10)
    seed: int | None = None


class EngineResult(BaseModel):
    engine_id: str
    status: str
    display_name: str | None = None
    capability_notes: list[str] = Field(default_factory=list)
    file: str | None = None
    url: str | None = None
    sample_rate: int | None = None
    duration_s: float | None = None
    elapsed_s: float | None = None
    error: str | None = None


class ComparisonJobParams(EngineRequest):
    job_name: str | None = None
    engine_ids: list[str]


class ComparisonJobDetail(BaseModel):
    job_name: str
    created_at: str
    updated_at: str
    engine_ids: list[str]
    status: str
    active_engine_id: str | None = None
    request: dict
    results: list[EngineResult] = Field(default_factory=list)


class ComparisonJobSummary(BaseModel):
    job_name: str
    created_at: str
    engine_ids: list[str]
    status: str
    result_count: int
    completed_count: int
    failed_count: int


@dataclass
class BaseTTSEngine:
    engine_id: str
    display_name: str
    capabilities: EngineCapabilities
    installed: bool = True
    install_hint: str | None = None
    load_error: str | None = None
    state: str = "unloaded"

    def get_status(self) -> EngineStatus:
        return EngineStatus(state=self.state, load_error=self.load_error, install_hint=self.install_hint)

    def validate_request(self, req: EngineRequest) -> str | None:
        return None

    async def generate(self, req: EngineRequest, job_name: str) -> EngineResult:
        raise NotImplementedError


class VoxCPMEngine(BaseTTSEngine):
    def __init__(self, model_state: ModelState):
        super().__init__("voxcpm", "VoxCPM2", EngineCapabilities(supports_voice_design=True, supports_cloning=True, supports_streaming=True))
        self.model_state = model_state

    async def generate(self, req: EngineRequest, job_name: str) -> EngineResult:
        apply_seed(req.seed)
        self.load_error = None
        self.state = "loading" if self.model_state.model is None else "currently_generating"
        try:
            model = await self.model_state.get_or_load()
        except Exception as exc:
            self.state = "error"
            self.load_error = sanitize_error(exc)
            raise
        self.state = "currently_generating"
        try:
            t0 = time.perf_counter()
            kwargs = build_generation_kwargs(
                text=assemble_prompt(req.text, req.voice_description),
                cfg_value=req.cfg_value,
                inference_timesteps=req.inference_timesteps,
                normalize=req.normalize,
                min_len=2,
                max_len=4096,
                retry_badcase=True,
                retry_badcase_max_times=5,
                retry_badcase_ratio_threshold=5.0,
                reference_wav_path=req.reference_wav_path,
            )
            async with self.model_state.lock:
                wav = await asyncio.to_thread(generate_with_retry, model, attempts=req.attempts, **kwargs)
            assert wav is not None
            elapsed = time.perf_counter() - t0
            filename = f"{safe_slug(job_name)}-{self.engine_id}-{timestamp_slug()}.wav"
            path = await asyncio.to_thread(save_wav_exact_or_unique, wav, filename, model.tts_model.sample_rate)
            self.load_error = None
            return EngineResult(engine_id=self.engine_id, display_name=self.display_name, status="completed", file=path.name, url=f"/outputs/{path.name}", sample_rate=model.tts_model.sample_rate, duration_s=len(wav)/model.tts_model.sample_rate, elapsed_s=elapsed)
        finally:
            if self.state != "error":
                self.state = "ready"


class LazyOptionalEngine(BaseTTSEngine):
    def __init__(self, engine_id: str, display_name: str, capabilities: EngineCapabilities, module_name: str):
        installed = module_is_installed(module_name)
        hint = None if installed else f"Install optional dependency for {display_name}"
        super().__init__(engine_id, display_name, capabilities, installed=False, install_hint=hint or f"Adapter for {display_name} is not implemented yet", state="not_installed")

    async def generate(self, req: EngineRequest, job_name: str) -> EngineResult:
        note = [f"{self.display_name} adapter is not implemented."]
        return EngineResult(engine_id=self.engine_id, display_name=self.display_name, status="failed", capability_notes=note, error="adapter_not_implemented")


class SupertonicEngine(BaseTTSEngine):
    def __init__(self):
        installed = module_is_installed("supertonic")
        hint = None if installed else "Install optional dependency: pip install supertonic"
        super().__init__("supertonic", "Supertonic", EngineCapabilities(supports_preset_voices=True), installed=installed, install_hint=hint, state="unloaded" if installed else "not_installed")

    async def generate(self, req: EngineRequest, job_name: str) -> EngineResult:
        self.load_error = None
        self.state = "downloading_or_initializing"
        try:
            TTS = importlib.import_module("supertonic").TTS
            tts = TTS(auto_download=True)
        except Exception as exc:
            self.state = "error"
            self.load_error = sanitize_error(exc)
            raise
        self.state = "currently_generating"
        try:
            t0 = time.perf_counter()
            style = tts.get_voice_style(voice_name="M1")
            wav, duration = await asyncio.to_thread(tts.synthesize, req.text, voice_style=style, lang="en")
            sample_rate = int(getattr(tts, "sample_rate", 24000))
            wav = normalize_wav_array(wav)
            duration_value = float(np.asarray(duration).reshape(-1)[0]) if np.asarray(duration).size else len(wav) / sample_rate
            filename = f"{safe_slug(job_name)}-{self.engine_id}-{timestamp_slug()}.wav"
            path = await asyncio.to_thread(save_wav_exact_or_unique, wav, filename, sample_rate)
            elapsed = time.perf_counter() - t0
            self.load_error = None
            return EngineResult(engine_id=self.engine_id, display_name=self.display_name, status="completed", file=path.name, url=f"/outputs/{path.name}", sample_rate=sample_rate, duration_s=duration_value, elapsed_s=elapsed)
        finally:
            if self.state != "error":
                self.state = "ready"


class KokoroEngine(BaseTTSEngine):
    def __init__(self):
        installed = module_is_installed("kokoro")
        hint = None if installed else "Install optional dependency: pip install kokoro; ensure espeak-ng is installed"
        super().__init__("kokoro", "Kokoro", EngineCapabilities(supports_preset_voices=True), installed=installed, install_hint=hint, state="unloaded" if installed else "not_installed")

    async def generate(self, req: EngineRequest, job_name: str) -> EngineResult:
        self.load_error = None
        self.state = "downloading_or_initializing"
        try:
            prepare_kokoro_runtime()
            KPipeline = importlib.import_module("kokoro").KPipeline
            pipeline = KPipeline(lang_code="a")
        except Exception as exc:
            self.state = "error"
            self.load_error = sanitize_error(exc)
            raise
        self.state = "currently_generating"
        try:
            t0 = time.perf_counter()

            def _generate():
                chunks = []
                for _gs, _ps, audio in pipeline(req.text, voice="af_heart"):
                    chunks.append(np.asarray(audio))
                if not chunks:
                    raise RuntimeError("Kokoro produced no audio")
                return np.concatenate(chunks)

            wav = await asyncio.to_thread(_generate)
            sample_rate = 24000
            filename = f"{safe_slug(job_name)}-{self.engine_id}-{timestamp_slug()}.wav"
            path = await asyncio.to_thread(save_wav_exact_or_unique, wav, filename, sample_rate)
            elapsed = time.perf_counter() - t0
            self.load_error = None
            return EngineResult(engine_id=self.engine_id, display_name=self.display_name, status="completed", file=path.name, url=f"/outputs/{path.name}", sample_rate=sample_rate, duration_s=len(wav) / sample_rate, elapsed_s=elapsed)
        finally:
            if self.state != "error":
                self.state = "ready"


class NeuTTSEngine(BaseTTSEngine):
    def __init__(self):
        installed = module_is_installed("neutts")
        hint = None if installed else "Install optional dependency: pip install neutts"
        super().__init__("neutts-nano", "NeuTTS Nano", EngineCapabilities(supports_plain_tts=False, supports_cloning=True, requires_reference_text=True), installed=installed, install_hint=hint, state="unloaded" if installed else "not_installed")

    async def generate(self, req: EngineRequest, job_name: str) -> EngineResult:
        self.load_error = None
        self.state = "downloading_or_initializing"
        try:
            NeuTTS = importlib.import_module("neutts").NeuTTS
            tts = NeuTTS(
                backbone_repo="neuphonic/neutts-nano",
                backbone_device="cpu",
                codec_repo="neuphonic/neucodec",
                codec_device="cpu",
            )
        except Exception as exc:
            self.state = "error"
            self.load_error = sanitize_error(exc)
            raise
        self.state = "currently_generating"
        try:
            t0 = time.perf_counter()

            def _generate():
                if req.reference_wav_path:
                    if not req.reference_text:
                        raise ValueError("NeuTTS Nano requires reference text for cloning")
                    ref_codes = tts.encode_reference(req.reference_wav_path)
                    return np.asarray(tts.infer(req.text, ref_codes, req.reference_text))
                return np.asarray(tts.infer(req.text))

            wav = await asyncio.to_thread(_generate)
            sample_rate = 24000
            filename = f"{safe_slug(job_name)}-{self.engine_id}-{timestamp_slug()}.wav"
            path = await asyncio.to_thread(save_wav_exact_or_unique, wav, filename, sample_rate)
            elapsed = time.perf_counter() - t0
            self.load_error = None
            return EngineResult(engine_id=self.engine_id, display_name=self.display_name, status="completed", file=path.name, url=f"/outputs/{path.name}", sample_rate=sample_rate, duration_s=len(wav) / sample_rate, elapsed_s=elapsed)
        finally:
            if self.state != "error":
                self.state = "ready"


class EngineManager:
    def __init__(self, model_state: ModelState):
        self.model_state = model_state
        self._job_lock = asyncio.Lock()
        self.engines: dict[str, BaseTTSEngine] = {
            "voxcpm": VoxCPMEngine(model_state),
            "supertonic": SupertonicEngine(),
            "neutts-nano": NeuTTSEngine(),
            "kokoro": KokoroEngine(),
        }

    def get(self, engine_id: str) -> BaseTTSEngine:
        engine_id = normalize_engine_id(engine_id)
        if engine_id not in self.engines:
            raise ValueError(f"Unknown engine: {engine_id}")
        return self.engines[engine_id]

    def list_engines(self) -> list[EngineInfo]:
        return [EngineInfo(engine_id=e.engine_id, display_name=e.display_name, installed=e.installed, status=e.get_status(), capabilities=e.capabilities) for e in self.engines.values()]

    def validate_selection(self, engine_ids: list[str]) -> list[BaseTTSEngine]:
        if not engine_ids:
            raise ValueError("engine_ids must not be empty")
        engines: list[BaseTTSEngine] = []
        for raw_engine_id in engine_ids:
            engine = self.get(raw_engine_id)
            if not engine.installed:
                raise ValueError(f"Engine not installed: {engine.engine_id}")
            engines.append(engine)
        return engines


def create_default_job_name() -> str:
    return f"job-{timestamp_slug()}"


def job_path(job_name: str) -> Path:
    return JOBS_DIR / f"{safe_slug(job_name)}.json"


def save_job_metadata(job: ComparisonJobDetail) -> None:
    job.updated_at = datetime.now().isoformat(timespec="seconds")
    _atomic_json_write(job_path(job.job_name), job.model_dump())


def list_jobs() -> list[ComparisonJobSummary]:
    ensure_jobs_dir()
    items = []
    for path in JOBS_DIR.glob("*.json"):
        data = _json_load(path)
        results = data.get("results", [])
        items.append(ComparisonJobSummary(
            job_name=data["job_name"], created_at=data["created_at"], engine_ids=data.get("engine_ids", []), status=data.get("status", "failed"), result_count=len(results),
            completed_count=sum(1 for r in results if r.get("status", "").startswith("completed")), failed_count=sum(1 for r in results if r.get("status") == "failed")
        ))
    return sorted(items, key=lambda j: j.created_at, reverse=True)


def get_job(job_name: str) -> ComparisonJobDetail:
    return ComparisonJobDetail(**_json_load(job_path(job_name)))


async def generate_comparison_job(manager: EngineManager, params: ComparisonJobParams) -> ComparisonJobDetail:
    engine_ids = [normalize_engine_id(e) for e in params.engine_ids]
    engines = manager.validate_selection(engine_ids)
    job_name = safe_slug(params.job_name or create_default_job_name())
    detail = ComparisonJobDetail(job_name=job_name, created_at=datetime.now().isoformat(timespec="seconds"), updated_at=datetime.now().isoformat(timespec="seconds"), engine_ids=engine_ids, status="in_progress", active_engine_id=None, request=params.model_dump(exclude={"engine_ids", "job_name"}), results=[])
    save_job_metadata(detail)
    async with manager._job_lock:
        for engine in engines:
            engine_id = engine.engine_id
            detail.active_engine_id = engine_id
            save_job_metadata(detail)
            req = EngineRequest(**params.model_dump(exclude={"engine_ids", "job_name"}))
            notes = []
            if req.voice_description and not engine.capabilities.supports_voice_design:
                notes.append(f"Generated plain TTS; voice design unsupported by {engine.display_name}.")
                req.voice_description = None
            if req.reference_wav_path and not engine.capabilities.supports_cloning:
                notes.append(f"Generated plain TTS; cloning unsupported by {engine.display_name}.")
                req.reference_wav_path = None
            if not req.reference_wav_path and not engine.capabilities.supports_plain_tts:
                result = EngineResult(engine_id=engine_id, display_name=engine.display_name, status="failed", capability_notes=notes, error=f"{engine.display_name} requires reference audio for generation")
                detail.results.append(result)
                save_job_metadata(detail)
                continue
            if req.reference_wav_path and engine.capabilities.supports_cloning and engine.capabilities.requires_reference_text and not params.reference_text:
                result = EngineResult(engine_id=engine_id, display_name=engine.display_name, status="failed", capability_notes=[f"{engine.display_name} requires reference text for cloning."], error="reference_text required")
                detail.results.append(result)
                save_job_metadata(detail)
                continue
            try:
                result = await engine.generate(req, job_name)
                result.capability_notes.extend(notes)
                if notes and result.status == "completed":
                    result.status = "completed_with_fallback"
            except Exception as exc:
                result = EngineResult(engine_id=engine_id, display_name=engine.display_name, status="failed", capability_notes=notes, error=sanitize_error(exc))
            finally:
                try:
                    if engine.state != "error":
                        engine.state = "ready"
                except Exception:
                    pass
            detail.results.append(result)
            save_job_metadata(detail)
    if any(r.status == "failed" for r in detail.results):
        detail.status = "completed_with_errors" if any(r.status.startswith("completed") for r in detail.results) else "failed"
    else:
        detail.status = "completed"
    detail.active_engine_id = None
    save_job_metadata(detail)
    return detail


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
    assert wav is not None

    elapsed = time.perf_counter() - t0
    path = await asyncio.to_thread(save_wav, wav, f"{mode}.wav", model.tts_model.sample_rate)
    return {
        "file": path.name,
        "url": f"/outputs/{path.name}",
        "sample_rate": model.tts_model.sample_rate,
        "duration_s": len(wav) / model.tts_model.sample_rate,
        "elapsed_s": elapsed,
    }
