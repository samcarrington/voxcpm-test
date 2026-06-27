---
goal: Add NeuTTS Nano as an optional local TTS and cloning engine through shared core abstractions
version: 1.0
date_created: 2026-06-26
owner: voxcpm-test maintainer
status: 'Planned'
tags: ['feature', 'architecture', 'tts', 'cloning', 'neutts', 'adapter']
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan adds NeuTTS Nano as an optional engine for local TTS and instant voice cloning. The design should extend a shared engine registry in `core.py` rather than adding separate paths inside `app.py` or `webapp.py`.

## Requirements & Constraints

- Preserve VoxCPM defaults and existing CLI/web behavior.
- Add engine capabilities metadata so UI/API can surface clone support.
- NeuTTS requires both reference audio and reference transcript for cloning.
- Batch mode should ship first; streaming is optional and backend-gated.
- Save all output through the existing wav/output pipeline.
- Keep dependency additions optional; do not hard-add heavy engine deps to base requirements.
- Source research: `docs/research/NEUTTS_NANO_RESEARCH.md`.

## Implementation Steps

### Phase 1: Core engine registry

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Add shared engine request/result/capability types in `core.py`. | | |
| TASK-002 | Register VoxCPM plus a neutral adapter interface for clone-capable engines. | | |
| TASK-003 | Add reference-audio and reference-text fields to shared request handling. | | |
| TASK-004 | Add optional engine dependency errors with install hints for `neutts`, backend extras, and streaming-only extras. | | |

### Phase 2: NeuTTS adapter

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Implement `NeuTTSEngine` using `NeuTTS(backbone_repo=..., backbone_device=..., codec_repo=..., codec_device=...)`. | | |
| TASK-006 | Support reference encoding via `encode_reference()` and cached reference codes. | | |
| TASK-007 | Require `reference_text` for clone requests and validate it in the request layer. | | |
| TASK-008 | Normalize output to 24000 Hz before handing off to `save_wav()`. | | |
| TASK-009 | Keep streaming disabled initially unless a GGUF backend and streaming dependencies are explicitly configured. | | |

### Phase 3: API and UI wiring

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | Add `--engine` CLI selection and preserve VoxCPM as the default. | | |
| TASK-011 | Add CLI/API support for `reference_text` alongside `reference_wav_path`. | | |
| TASK-012 | Add an engine listing endpoint for UI discovery. | | |
| TASK-013 | Surface clone-specific fields in the web form only when the chosen engine supports them. | | |

## Verification

- Import and registry smoke tests
- Optional clone generation smoke after installing NeuTTS deps
- Optional streaming smoke only when GGUF/pyaudio path is present
- Negative validation smoke: clone request with reference audio but no transcript must fail with a clear message
