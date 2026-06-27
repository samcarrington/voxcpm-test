---
goal: Add Supertonic as an optional local TTS engine without changing VoxCPM defaults
version: 1.0
date_created: 2026-06-26
owner: voxcpm-test maintainer
status: 'Planned'
tags: ['feature', 'architecture', 'tts', 'supertonic', 'adapter']
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan adds Supertonic as an optional engine behind the existing app architecture. The default VoxCPM path remains unchanged. Implementation should introduce a small multi-engine abstraction in `core.py` first, then add a Supertonic adapter that plugs into CLI and web APIs.

## Requirements & Constraints

- Preserve VoxCPM default behavior and output conventions.
- Add an engine registry in `core.py` before engine-specific code lands.
- Keep `app.py` and `webapp.py` thin; they should select engines via shared core helpers.
- Supertonic is preset-voice batch TTS only; do not present it as reference-audio cloning.
- Use existing `save_wav()` and `output/` conventions for all final audio.
- Do not force Supertonic into base `requirements.txt`; prefer optional deps or a separate engine requirements file.
- Include engine capability metadata so the UI can hide unsupported clone/stream actions.
- Source research: `docs/research/SUPERTONIC_RESEARCH.md`.

## Implementation Steps

### Phase 1: Engine abstraction in core

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Add `EngineCapabilities`, `EngineRequest`, `EngineResult`, and a `BaseTTSEngine` protocol or abstract base in `core.py`. | | |
| TASK-002 | Add an engine registry keyed by `engine_id`, with VoxCPM as the default registered engine. | | |
| TASK-003 | Add shared helpers for resolving engine capabilities and generating final WAVs through the selected adapter. | | |
| TASK-004 | Add optional-dependency error handling so selecting an uninstalled engine returns a clear install hint instead of crashing import-time. | | |

### Phase 2: Supertonic adapter

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Implement a `SupertonicEngine` adapter using `from supertonic import TTS`. | | |
| TASK-006 | Map preset voice selection to supported built-in IDs (`M1`-`M5`, `F1`-`F5`) and optionally accept a validated style JSON path. | | |
| TASK-007 | Discover and normalize output sample rate from the SDK path before saving with `save_wav()`. | | |
| TASK-008 | Mark the adapter as batch-only, non-cloning, non-streaming, preset-voice capable in capability metadata. | | |

### Phase 3: CLI and API wiring

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-009 | Add `--engine` to the CLI with VoxCPM as the default and add a Supertonic voice/style selector flag. | | |
| TASK-010 | Add `/api/engines` or equivalent listing endpoint in the web app. | | |
| TASK-011 | Ensure request routing uses shared core engine selection logic, not bespoke `app.py` branches. | | |
| TASK-012 | Update the SPA to disable reference upload and streaming controls when Supertonic is selected. | | |

## Verification

- Import check for `core` with the registry loaded
- Smoke test selecting Supertonic without touching VoxCPM defaults
- Manual batch generation after installing optional Supertonic deps
- Confirm selecting Supertonic without `supertonic` installed returns the documented install hint
