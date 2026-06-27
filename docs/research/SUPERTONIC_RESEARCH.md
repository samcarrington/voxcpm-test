# Supertonic Research Report

**Research Date:** 2026-06-26  
**Issue:** #2  
**Issue:** https://github.com/samcarrington/voxcpm-test/issues/2  
**Repo:** https://github.com/supertone-inc/supertonic  
**SDK:** https://github.com/supertone-inc/supertonic-py  
**Model:** https://huggingface.co/Supertone/supertonic-3  
**License:** SDK/repo MIT; model OpenRAIL-M

## Overview

Supertonic is a local TTS engine with preset voices and voice-style files. The open package is batch-first and does not advertise on-device arbitrary reference-audio cloning.

For this repo, Supertonic is useful as a fast local comparison point against VoxCPM's larger, clone-capable workflow. It should be presented as preset/style TTS, not as a drop-in replacement for VoxCPM clone mode.

## Install / Runtime

- `pip install supertonic`
- Optional: `supertonic[serve]`
- Python >= 3.9
- ONNX Runtime local inference
- First run downloads ~400 MB assets
- Public quickstart does not document explicit `cpu`/`cuda`/`mps` device selection; assume SDK-managed runtime until verified locally

## API Sketch

```python
from supertonic import TTS

tts = TTS(auto_download=True)
style = tts.get_voice_style(voice_name="M1")
wav, duration = tts.synthesize("Hello world", voice_style=style, lang="en")
tts.save_audio(wav, "output.wav")
```

## Capabilities

- Local TTS
- Built-in voices M1-M5 and F1-F5
- Custom voice-style JSONs from Voice Builder
- Batch synthesis
- Vendor-hosted products expose additional voice tooling; the open package should only rely on locally available style files

## Streaming

- No clear first-class Python streaming API found
- Treat as final-WAV only

## Caveats

- No official open-model arbitrary reference-audio cloning
- CPU-oriented guidance; no explicit CPU/CUDA/MPS selection in quickstart
- Sample rate is not clearly stated in quickstart; SDK handles save_audio
- Adapter must discover the waveform sample rate or use a verified SDK constant before calling this repo's `save_wav()`

## Integration Implications

- Best fit for a preset-voice engine adapter
- Add an explicit `voice_id`/`style_path` field instead of overloading VoxCPM's free-form `voice_description`
- Do not expose clone-mode semantics as equivalent to VoxCPM cloning
- Add adapter logic for batch-only generation; prefer returning the waveform to the shared save path rather than using SDK-managed output filenames
- Mark capabilities as `supports_cloning=False`, `supports_streaming=False`, `supports_voice_description=False`, `supports_preset_voices=True`

## Sources

- https://github.com/supertone-inc/supertonic
- https://github.com/supertone-inc/supertonic-py
- https://huggingface.co/Supertone/supertonic-3
- https://supertone-inc.github.io/supertonic-py/quickstart/
- https://pypi.org/project/supertonic/
