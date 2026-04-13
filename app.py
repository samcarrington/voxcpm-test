"""
VoxCPM2 Text-to-Speech Test App
================================
Demonstrates voice design, voice cloning, and streaming generation
using the VoxCPM2 model from OpenBMB via HuggingFace.

Usage:
    # Basic voice design (no reference audio needed)
    python app.py

    # Voice cloning from a reference file
    python app.py --clone --reference speaker.wav

    # Streaming generation
    python app.py --stream

    # Custom text
    python app.py --text "Hello, world!"

    # List available devices
    python app.py --info
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output")

# More conservative defaults for better quality/consistency.
DEFAULT_CFG = 2.0
DEFAULT_STEPS = 20
STREAMING_STEPS = 12


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def detect_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    effective_device = getattr(model.tts_model, "device", "unknown")

    elapsed = time.perf_counter() - t0
    print(f"Model loaded in {elapsed:.1f}s (runtime device={effective_device})")
    return model


def save_wav(wav: np.ndarray, filename: str, sample_rate: int = 48000):
    """Save a numpy waveform to a WAV file in the output directory."""
    ensure_output_dir()
    path = OUTPUT_DIR / filename
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


# ---------------------------------------------------------------------------
# Generation modes
# ---------------------------------------------------------------------------


def demo_voice_design(
    model,
    text: str | None = None,
    *,
    cfg_value: float = DEFAULT_CFG,
    inference_timesteps: int = DEFAULT_STEPS,
    normalize: bool = True,
    attempts: int = 2,
    min_len: int = 2,
    max_len: int = 4096,
    retry_badcase: bool = True,
    retry_badcase_max_times: int = 5,
    retry_badcase_ratio_threshold: float = 5.0,
):
    """
    Voice Design mode: generate speech from a text description of the
    desired voice, without any reference audio.
    """
    print("\n--- Voice Design Demo ---")

    voices = [
        {
            "description": "Calm English female voice",
            "text": text
            or "The Punters' Club; Good music from tolerable hosts, with disco shoulders and glittery jackets. Join us for a boogie.",
        },
        {
            "description": "Calm English male voice",
            "text": text
            or "Have you got a disco leg and a desire to shake it? You're in the Punters' Club, on Radio Waters",
        },
    ]

    results = []
    for i, v in enumerate(voices):
        prompt = f"({v['description']}){v['text']}"
        print(f"\n[{i + 1}] Voice: {v['description']}")
        print(f"    Text:  {v['text']}")

        t0 = time.perf_counter()
        gen_kwargs = build_generation_kwargs(
            text=prompt,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            min_len=min_len,
            max_len=max_len,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        )
        wav = generate_with_retry(model, attempts=attempts, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        print(f"    Generated in {elapsed:.1f}s")

        filename = f"voice_design_{i + 1}.wav"
        save_wav(wav, filename, model.tts_model.sample_rate)
        results.append(filename)

    return results


def demo_voice_clone(
    model,
    reference_path: str,
    text: str | None = None,
    *,
    cfg_value: float = DEFAULT_CFG,
    inference_timesteps: int = DEFAULT_STEPS,
    normalize: bool = True,
    attempts: int = 2,
    min_len: int = 2,
    max_len: int = 4096,
    retry_badcase: bool = True,
    retry_badcase_max_times: int = 5,
    retry_badcase_ratio_threshold: float = 5.0,
):
    """
    Voice Cloning mode: clone a speaker's voice from a reference audio file.
    VoxCPM2 supports reference-only cloning (no transcript needed).
    """
    print("\n--- Voice Cloning Demo ---")

    if not os.path.isfile(reference_path):
        print(f"Error: reference audio not found: {reference_path}")
        sys.exit(1)

    target_text = text or "This is a demonstration of voice cloning with VoxCPM2."
    print(f"Reference: {reference_path}")
    print(f"Text:      {target_text}")

    t0 = time.perf_counter()
    gen_kwargs = build_generation_kwargs(
        text=target_text,
        reference_wav_path=reference_path,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
        min_len=min_len,
        max_len=max_len,
        retry_badcase=retry_badcase,
        retry_badcase_max_times=retry_badcase_max_times,
        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
    )
    wav = generate_with_retry(model, attempts=attempts, **gen_kwargs)
    elapsed = time.perf_counter() - t0
    print(f"Generated in {elapsed:.1f}s")

    return save_wav(wav, "voice_clone.wav", model.tts_model.sample_rate)


def demo_streaming(
    model,
    text: str | None = None,
    *,
    cfg_value: float = DEFAULT_CFG,
    inference_timesteps: int = STREAMING_STEPS,
    normalize: bool = True,
    min_len: int = 2,
    max_len: int = 4096,
):
    """
    Streaming mode: generate audio incrementally, useful for real-time playback.
    """
    print("\n--- Streaming Demo ---")

    target_text = text or (
        "Streaming generation allows you to begin playback before the "
        "entire utterance has been synthesized. This is especially useful "
        "for interactive applications and voice assistants."
    )
    print(f"Text: {target_text}")

    t0 = time.perf_counter()
    chunks = []
    for i, chunk in enumerate(
        model.generate_streaming(
            text=target_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            min_len=min_len,
            max_len=max_len,
            retry_badcase=False,
        )
    ):
        chunks.append(chunk)
        chunk_dur = len(chunk) / model.tts_model.sample_rate
        print(f"  Chunk {i + 1}: {chunk_dur:.2f}s ({len(chunk)} samples)")

    wav = np.concatenate(chunks)
    elapsed = time.perf_counter() - t0
    print(f"Total generation: {elapsed:.1f}s")

    return save_wav(wav, "streaming_output.wav", model.tts_model.sample_rate)


def show_info():
    """Print environment info for debugging."""
    import torch

    print("=== Environment Info ===")
    print(f"Python:     {sys.version}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.cuda.is_available()} ({torch.version.cuda or 'N/A'})")
    print(f"MPS:        {torch.backends.mps.is_available()}")
    print(f"Device:     {detect_device()}")

    try:
        from importlib.metadata import version

        print(f"VoxCPM:     {version('voxcpm')}")
    except Exception:
        print("VoxCPM:     not installed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="VoxCPM2 Text-to-Speech test app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--text", "-t", type=str, default=None, help="Custom text to synthesize"
    )
    p.add_argument("--clone", action="store_true", help="Run voice cloning demo")
    p.add_argument(
        "--reference",
        "-r",
        type=str,
        default=None,
        help="Path to reference audio for cloning",
    )
    p.add_argument(
        "--stream", action="store_true", help="Run streaming generation demo"
    )
    p.add_argument(
        "--denoiser",
        action="store_true",
        help="Load the ZipEnhancer denoiser (slower startup)",
    )
    p.add_argument(
        "--info", action="store_true", help="Print environment info and exit"
    )
    p.add_argument(
        "--all", action="store_true", help="Run all demos (design + streaming)"
    )
    p.add_argument(
        "--cfg",
        type=float,
        default=DEFAULT_CFG,
        help=f"CFG guidance value (default: {DEFAULT_CFG})",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Inference timesteps for non-streaming demos (default: {DEFAULT_STEPS})",
    )
    p.add_argument(
        "--stream-steps",
        type=int,
        default=STREAMING_STEPS,
        help=f"Inference timesteps for streaming demo (default: {STREAMING_STEPS})",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization before synthesis",
    )
    p.add_argument(
        "--attempts",
        type=int,
        default=2,
        help="Retry attempts for non-streaming demos (default: 2)",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Minimum generated length in model units (default: 2)",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=4096,
        help="Maximum generated length in model units (default: 4096)",
    )
    p.add_argument(
        "--retry-badcase-max-times",
        type=int,
        default=5,
        help="VoxCPM internal badcase retry max times (default: 5)",
    )
    p.add_argument(
        "--retry-badcase-ratio-threshold",
        type=float,
        default=5.0,
        help="VoxCPM badcase audio/text ratio threshold (default: 5.0)",
    )
    p.add_argument(
        "--disable-badcase-retry",
        action="store_true",
        help="Disable VoxCPM internal badcase retry loop",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional torch random seed for reproducibility",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.info:
        show_info()
        return

    if args.seed is not None:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using seed: {args.seed}")

    model = load_model(load_denoiser=args.denoiser)
    normalize = not args.no_normalize
    retry_badcase = not args.disable_badcase_retry

    ran_something = False

    # Voice design (default if nothing else selected)
    if args.all or (not args.clone and not args.stream):
        demo_voice_design(
            model,
            text=args.text,
            cfg_value=args.cfg,
            inference_timesteps=args.steps,
            normalize=normalize,
            attempts=max(1, args.attempts),
            min_len=max(1, args.min_len),
            max_len=max(32, args.max_len),
            retry_badcase=retry_badcase,
            retry_badcase_max_times=max(1, args.retry_badcase_max_times),
            retry_badcase_ratio_threshold=max(1.5, args.retry_badcase_ratio_threshold),
        )
        ran_something = True

    # Voice cloning
    if args.clone:
        if not args.reference:
            print("Error: --clone requires --reference <audio_file>")
            sys.exit(1)
        demo_voice_clone(
            model,
            args.reference,
            text=args.text,
            cfg_value=args.cfg,
            inference_timesteps=args.steps,
            normalize=normalize,
            attempts=max(1, args.attempts),
            min_len=max(1, args.min_len),
            max_len=max(32, args.max_len),
            retry_badcase=retry_badcase,
            retry_badcase_max_times=max(1, args.retry_badcase_max_times),
            retry_badcase_ratio_threshold=max(1.5, args.retry_badcase_ratio_threshold),
        )
        ran_something = True

    # Streaming
    if args.stream or args.all:
        demo_streaming(
            model,
            text=args.text,
            cfg_value=args.cfg,
            inference_timesteps=args.stream_steps,
            normalize=normalize,
            min_len=max(1, args.min_len),
            max_len=max(32, args.max_len),
        )
        ran_something = True

    if ran_something:
        print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
