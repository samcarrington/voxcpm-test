# Kokoro Research Report

**Research Date:** 2026-06-26  
**Issue:** #4  
**Issue:** https://github.com/samcarrington/voxcpm-test/issues/4  
**Repo:** https://github.com/hexgrad/kokoro  
**Model:** https://huggingface.co/hexgrad/Kokoro-82M  
**License:** Apache-2.0

## Overview

Kokoro is a thin, small-footprint local TTS engine with preset voices. It is simple to integrate but does not provide general reference-audio cloning.

For this repo, Kokoro is valuable as a small, fast preset-voice baseline. It should be exposed as a text + language + preset voice workflow, separate from VoxCPM's voice-design and reference-audio clone paths.

## Install / Runtime

- `pip install kokoro`
- Example environments may need `espeak-ng` for phonemizer/G2P
- Small 82M model
- 24 kHz output
- Public examples use `KPipeline(lang_code=...)`; device behavior should be verified during implementation rather than assumed

## API Sketch

```python
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')
generator = pipeline("Hello world", voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f"{i}.wav", audio, 24000)
```

## Capabilities

- Local TTS
- Preset voicepacks
- Chunked generator output
- Lightweight model
- Generator returns grapheme/phoneme metadata alongside audio chunks, which can be useful for debug logging

## Streaming

- Generator yields chunks of graphemes, phonemes, and audio
- Not a formal streaming API, but can be adapted into pseudo-streaming

## Caveats

- No general reference-audio cloning
- `espeak-ng` system dependency may be required
- Voice selection is preset-based rather than arbitrary style cloning
- Language code and voice preset need validation before generation to keep errors user-friendly

## Integration Implications

- Best fit for a minimal preset-voice engine adapter
- Map current text-only generation cleanly
- Treat chunk output as batch or pseudo-streaming, not true clone streaming
- Mark capabilities as `supports_cloning=False`, `supports_preset_voices=True`, `sample_rate=24000`, `supports_streaming=False` or `pseudo_streaming=True` depending on adapter design

## Sources

- https://github.com/hexgrad/kokoro
- https://huggingface.co/hexgrad/Kokoro-82M
