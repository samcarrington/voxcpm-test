# NeuTTS Nano Research Report

**Research Date:** 2026-06-26  
**Issue:** #3  
**Issue:** https://github.com/samcarrington/voxcpm-test/issues/3  
**Library:** https://github.com/neuphonic/neutts  
**Model:** https://huggingface.co/neuphonic/neutts-nano  
**License:** NeuTTS Open License 1.0

## Overview

NeuTTS Nano is a local TTS + instant voice cloning stack. It supports reference audio plus reference text and is the strongest match for cloning-oriented integrations.

For this repo, NeuTTS Nano is the closest conceptual match to VoxCPM clone mode, but its clone contract differs: it requires both the reference audio and the reference transcript. The UI/API should make that explicit instead of silently reusing the existing `reference_wav_path`-only flow.

## Install / Runtime

- `pip install neutts`
- Optional extras: `neutts[all]`, `neutts[llama]`
- Streaming examples require `pyaudio`
- GGUF backbones may use `llama-cpp-python`
- ONNX decoder path uses `onnxruntime`
- Device is explicit in public examples via `backbone_device` and `codec_device`

## API Sketch

```python
from neutts import NeuTTS

tts = NeuTTS(
    backbone_repo="neuphonic/neutts-nano",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec",
    codec_device="cpu",
)
ref_text = open("samples/jo.txt").read().strip()
ref_codes = tts.encode_reference("samples/jo.wav")
wav = tts.infer("Hello world", ref_codes, ref_text)
```

```python
import soundfile as sf
sf.write("test.wav", wav, 24000)
```

## Capabilities

- Local TTS
- Instant voice cloning
- Reference pre-encoding to reduce latency
- Explicit backbone/codec device selection
- 24 kHz output
- Reference audio can be encoded once and cached as speaker/reference codes

## Streaming

- Streaming exists
- Current docs indicate GGUF backbones only
- Requires `pyaudio`

## Caveats

- Clone mode requires both reference audio and transcript
- Batch and streaming paths are not equally mature
- Device and decoder/backend dependencies vary by install mode
- Streaming should not be promised in the generic UI until the selected NeuTTS backend reports support

## Integration Implications

- Needs a request shape that can carry `reference_audio` and `reference_text`
- Add UI/API validation for clone inputs
- Keep batch synthesis as the default path first
- Gate streaming behind backend capability checks
- Mark capabilities as `supports_cloning=True`, `requires_reference_text=True`, `sample_rate=24000`, with streaming conditional on GGUF backend setup

## Sources

- https://github.com/neuphonic/neutts
- https://huggingface.co/neuphonic/neutts-nano
- https://huggingface.co/neuphonic/neutts-nano/blob/main/README.md
