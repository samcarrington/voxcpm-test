"""VoxCPM2 TTS CLI."""

import argparse
import asyncio
import time
import sys

import numpy as np

from core import (
    DEFAULT_CFG,
    DEFAULT_STEPS,
    STREAMING_STEPS,
    EngineManager,
    EngineRequest,
    GenerationParams,
    ModelState,
    detect_nvidia_gpus,
    print_cuda_diagnosis,
    show_info,
    normalize_engine_id,
    generate_with_retry,
    build_generation_kwargs,
    load_model,
    save_wav,
    OUTPUT_DIR,
)


OPTIONAL_ENGINE_IDS = {"supertonic", "neutts-nano", "kokoro"}


def parse_args():
    p = argparse.ArgumentParser(description="VoxCPM2 Text-to-Speech test app")
    p.add_argument("--text", "-t", type=str, default=None)
    p.add_argument("--clone", action="store_true")
    p.add_argument("--reference", "-r", type=str, default=None)
    p.add_argument("--reference-text", type=str, default=None)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--denoiser", action="store_true")
    p.add_argument("--info", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--engine", type=str, default="voxcpm")
    p.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--stream-steps", type=int, default=STREAMING_STEPS)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--retry-badcase-max-times", type=int, default=5)
    p.add_argument("--retry-badcase-ratio-threshold", type=float, default=5.0)
    p.add_argument("--disable-badcase-retry", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def validate_cli(args, manager: EngineManager):
    engine_id = normalize_engine_id(args.engine)
    engine = manager.get(engine_id)
    if args.clone and not engine.capabilities.supports_cloning:
        raise SystemExit(f"Error: {engine.display_name} does not support cloning")
    if args.stream and not engine.capabilities.supports_streaming:
        raise SystemExit(f"Error: {engine.display_name} does not support streaming")
    if args.clone and engine.capabilities.requires_reference_text and not args.reference_text:
        raise SystemExit(f"Error: {engine.display_name} requires --reference-text for cloning")
    if args.clone and not args.reference:
        raise SystemExit("Error: --clone requires --reference <audio_file>")
    if not engine.installed:
        raise SystemExit(f"Error: {engine.display_name} is not installed: {engine.install_hint}")


def main():
    args = parse_args()
    if args.info:
        show_info()
        return
    print_cuda_diagnosis()
    manager = EngineManager(ModelState())
    validate_cli(args, manager)

    if args.engine != "voxcpm":
        engine = manager.get(normalize_engine_id(args.engine))
        request = EngineRequest(
            text=args.text or "Hello",
            reference_wav_path=args.reference,
            reference_text=args.reference_text,
            cfg_value=args.cfg,
            inference_timesteps=args.steps,
            normalize=not args.no_normalize,
            attempts=max(1, args.attempts),
            seed=args.seed,
        )
        result = asyncio.run(engine.generate(request, "cli"))
        if result.status == "failed":
            raise SystemExit(f"Error: {result.error or 'generation failed'}")
        print(f"Saved outputs to {OUTPUT_DIR.resolve()}")
        return

    model = load_model(load_denoiser=args.denoiser)

    if args.all or (not args.clone and not args.stream):
        result = generate_with_retry(model, attempts=max(1, args.attempts), **build_generation_kwargs(text=args.text or "Hello", cfg_value=args.cfg, inference_timesteps=args.steps, normalize=not args.no_normalize, min_len=max(1, args.min_len), max_len=max(32, args.max_len), retry_badcase=not args.disable_badcase_retry, retry_badcase_max_times=max(1, args.retry_badcase_max_times), retry_badcase_ratio_threshold=max(1.5, args.retry_badcase_ratio_threshold)))
        if result is None:
            raise SystemExit("Error: generation failed")
        save_wav(result, "cli.wav", model.tts_model.sample_rate)

    if args.stream or args.all:
        target_text = args.text or (
            "Streaming generation allows you to begin playback before the "
            "entire utterance has been synthesized."
        )
        print("\n--- Streaming Demo ---")
        print(f"Text: {target_text}")
        chunks = []
        t0 = time.perf_counter()
        for i, chunk in enumerate(
            model.generate_streaming(
                text=target_text,
                cfg_value=args.cfg,
                inference_timesteps=args.stream_steps,
                normalize=not args.no_normalize,
                min_len=max(1, args.min_len),
                max_len=max(32, args.max_len),
                retry_badcase=False,
            )
        ):
            chunks.append(chunk)
            chunk_dur = len(chunk) / model.tts_model.sample_rate
            print(f"  Chunk {i + 1}: {chunk_dur:.2f}s ({len(chunk)} samples)")
        if not chunks:
            raise SystemExit("Error: streaming produced no audio")
        wav = np.concatenate(chunks)
        elapsed = time.perf_counter() - t0
        print(f"Total generation: {elapsed:.1f}s")
        save_wav(wav, "streaming_output.wav", model.tts_model.sample_rate)

    if args.clone:
        kwargs = build_generation_kwargs(
            text=args.text or "This is a demonstration of voice cloning with VoxCPM2.",
            cfg_value=args.cfg,
            inference_timesteps=args.steps,
            normalize=not args.no_normalize,
            min_len=max(1, args.min_len),
            max_len=max(32, args.max_len),
            retry_badcase=not args.disable_badcase_retry,
            retry_badcase_max_times=max(1, args.retry_badcase_max_times),
            retry_badcase_ratio_threshold=max(1.5, args.retry_badcase_ratio_threshold),
            reference_wav_path=args.reference,
        )
        result = generate_with_retry(model, attempts=max(1, args.attempts), **kwargs)
        if result is None:
            raise SystemExit("Error: generation failed")
        save_wav(result, "voice_clone.wav", model.tts_model.sample_rate)

    print(f"Saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
