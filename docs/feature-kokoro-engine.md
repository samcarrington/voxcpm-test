---
goal: Add Kokoro as an optional lightweight local TTS engine with preset voices
version: 1.0
date_created: 2026-06-26
owner: voxcpm-test maintainer
status: 'Planned'
tags: ['feature', 'architecture', 'tts', 'kokoro', 'adapter']
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan adds Kokoro as a lightweight preset-voice engine. It should integrate through the same shared engine abstraction used for other optional engines, while leaving VoxCPM as the default path.

## Requirements & Constraints

- Preserve current VoxCPM CLI/web defaults.
- Add engine registration in `core.py` before wiring UI or CLI selection.
- Kokoro is preset-voice only; do not model it as a clone engine.
- Treat chunk output as batch or pseudo-streaming, not guaranteed live streaming.
- Keep `save_wav()` and `output/` as the final persistence path.
- Note the `espeak-ng` system dependency in docs and setup guidance, not base requirements.
- Source research: `docs/research/KOKORO_RESEARCH.md`.

## Implementation Steps

### Phase 1: Shared abstraction

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Add engine capability types and registry support in `core.py`. | | |
| TASK-002 | Ensure engine selection works across CLI and web helpers without branching duplication. | | |
| TASK-003 | Add optional-dependency error handling so missing `kokoro` or `espeak-ng` guidance is surfaced clearly. | | |

### Phase 2: Kokoro adapter

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-004 | Implement `KokoroEngine` using `KPipeline`. | | |
| TASK-005 | Map language and preset voice selection into adapter parameters. | | |
| TASK-006 | Validate `lang_code` and voice preset early where possible. | | |
| TASK-007 | Consume generator chunks and assemble final audio for `save_wav()` at 24000 Hz. | | |
| TASK-008 | Mark the engine as non-cloning and optionally pseudo-streamable only. | | |

### Phase 3: Surface area updates

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-009 | Add `--engine` selection to the CLI while preserving VoxCPM default behavior. | | |
| TASK-010 | Add `--voice` and `--lang-code` controls for Kokoro usage. | | |
| TASK-011 | Add an engine list endpoint for the web app. | | |
| TASK-012 | Document `espeak-ng` setup notes and voice limitations in the README or engine docs. | | |

## Verification

- Import and registry smoke checks
- Manual generation smoke with Kokoro installed
- Confirm outputs are saved through existing naming conventions
- Negative UI/API smoke: reference upload controls are not offered when Kokoro is selected
