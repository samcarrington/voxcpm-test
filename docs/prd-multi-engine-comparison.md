# PRD: Multi-Engine TTS Comparison Jobs

**Date:** 2026-06-27  
**Owner:** voxcpm-test maintainer  
**Status:** Draft  
**Related issues:** #2 Supertonic, #3 Neuphonic NeuTTS Nano, #4 Kokoro  
**Related research:**

- `docs/research/SUPERTONIC_RESEARCH.md`
- `docs/research/NEUTTS_NANO_RESEARCH.md`
- `docs/research/KOKORO_RESEARCH.md`

---

## 1. Executive Summary

### Problem Statement

The app currently generates audio through one model workflow at a time, which makes direct comparison across VoxCPM, Supertonic, NeuTTS Nano, and Kokoro slow and hard to correlate. Generated files also lack a shared job identifier and model identifier, so comparing outputs after several runs is error-prone.

### Proposed Solution

Add a web/API comparison job workflow that lets the user enter text once, select one, several, or all available TTS engines, and generate sequential outputs grouped under one shared job name. Each output must record the engine used, preserve a common job identifier, expose model load/download/init status in the top bar, and make capability fallbacks explicit when selected engines cannot perform voice design or cloning.

### Success Criteria

- A user can generate the same text across all available engines from one web form submission.
- A user can select any subset of engines, including exactly one engine, for a comparison job.
- Every generated WAV filename includes the shared job name, engine ID, and timestamp.
- The web UI groups all outputs from the same job together without relying on filename parsing alone.
- Unsupported voice design or cloning requests fall back to plain supported TTS and show a clear per-engine result note.
- Model loading state for each selected engine is visible before and during first use.
- Comparison jobs are final-WAV only in MVP; existing streaming remains a single-engine action.

---

## 2. User Experience & Functionality

### User Personas

- **Solo TTS evaluator:** A developer comparing local TTS engines for quality, latency, cloning behavior, and subjective voice fit.
- **Model integration maintainer:** A developer adding and debugging optional TTS engines while preserving the current VoxCPM default workflow.

### Primary User Flow

1. User opens the local web app.
2. Top bar shows available engines and their load status: not installed, unloaded, loading/downloading, ready, or error.
3. User enters the text to synthesize.
4. User optionally enters voice-design instructions and/or selects reference audio for cloning.
5. User selects one, multiple, or all engines.
6. App auto-generates an editable job name, for example `job-20260627-143012`.
7. User starts the comparison job.
8. App runs selected engines sequentially.
9. Each engine result appears under the shared job group as it completes.
10. Unsupported capabilities fall back to the closest supported plain TTS path and show a visible note.
11. User can play/download each output and compare all outputs in the grouped result set.

### User Stories

#### Story 1: Multi-engine comparison

As a TTS evaluator, I want to generate the same text across multiple selected models so that I can compare audio quality without manually repeating the same prompt.

**Acceptance Criteria**

- The web UI must provide engine selection controls with options for one, many, or all installed/available engines.
- The API must accept an ordered list of engine IDs for a comparison job.
- The app must run selected engines sequentially by default.
- Results must appear grouped under one job name.
- A failed engine must not erase completed outputs from other engines in the same job.

#### Story 2: Correlated result files

As a TTS evaluator, I want every output filename to include the job and model so that files remain traceable outside the UI.

**Acceptance Criteria**

- Each comparison job must have one shared `job_name`.
- `job_name` must be auto-generated before submission and editable by the user.
- Generated filenames must follow this pattern:

  ```text
  <job-name>-<engine-id>-<timestamp>.wav
  ```

- `job_name` and `engine_id` must be sanitized for filesystem safety.
- The API response for each output must include `job_name`, `engine_id`, `file`, `url`, `sample_rate`, `duration_s`, `elapsed_s`, and `status`.

#### Story 3: Capability-aware fallback

As a TTS evaluator, I want the app to tell me when a model cannot perform voice design or cloning so that I do not misinterpret plain TTS as a styled or cloned result.

**Acceptance Criteria**

- Engine capability metadata must identify whether an engine supports:
  - plain text-to-speech
  - voice design / free-form style prompting
  - reference-audio cloning
  - required reference transcript
  - preset voices
  - final WAV generation
  - streaming
- If the user requests voice design and an engine does not support it, that engine must still generate plain supported TTS.
- If the user requests cloning and an engine does not support it, that engine must still generate plain supported TTS.
- If an engine supports cloning but required clone inputs are missing, that engine must not silently fall back. It must return a failed result with a clear validation error.
- The result row must show a capability note, for example:
  - `Generated plain TTS; voice design unsupported by Kokoro.`
  - `Generated plain TTS; cloning unsupported by Supertonic.`
- Missing required clone inputs must show an error note, for example:
  - `NeuTTS Nano requires reference text for cloning.`
- The UI must not silently discard user intent.

#### Story 4: Visible model load status

As a TTS evaluator, I want model load/download/init status in the top bar so that I understand why first generation may be slow.

**Acceptance Criteria**

- The top bar must show one status indicator per engine.
- Supported states must include at least:
  - `not_installed`
  - `unloaded`
  - `loading`
  - `downloading_or_initializing`
  - `ready`
  - `error`
- During a comparison job, the currently active engine must be visually distinguishable.
- If an optional engine dependency is missing, its status must show an install hint rather than a generic failure.
- The status endpoint must be pollable without triggering model load.

#### Story 5: CLI single-engine behavior

As a maintainer, I want the CLI to remain single-engine but selectable at runtime so that command-line usage stays simple and deterministic.

**Acceptance Criteria**

- CLI must support selecting one engine at runtime, for example `--engine voxcpm` or `--engine kokoro`.
- CLI must not run multi-engine comparison jobs.
- CLI must error before generation when selected capability options are incompatible with the selected engine.
- CLI must provide `--reference-text` for engines that require a transcript for clone mode.
- Examples:
  - `--engine kokoro --clone` must fail with a clear message because Kokoro does not support reference-audio cloning.
  - `--engine supertonic --stream` must fail if Supertonic streaming is unsupported.
  - `--engine neutts --clone` must require reference audio and reference text.

### Non-Goals

- No concurrent multi-model generation in the MVP.
- No scoring/ranking of generated audio quality.
- No persisted database for jobs; filesystem-backed metadata or JSON sidecars are acceptable for MVP.
- No cloud deployment or multi-user support.
- No guarantee that every engine supports streaming.
- No voice cloning implementation for engines whose local model does not support it.
- No CLI multi-engine batch/comparison mode.
- No multi-engine streaming comparison in MVP.

---

## 3. AI System Requirements

### Tool Requirements

- Local TTS engine adapters for:
  - VoxCPM using `engine_id=voxcpm` and model `openbmb/VoxCPM2`
  - Supertonic
  - NeuTTS Nano
  - Kokoro
- Engine registry that exposes installed status, load status, and capabilities.
- Shared generation request type that can carry common fields plus engine-specific optional fields.
- Output persistence that writes WAV files to the existing `output/` directory.

### MVP Engine Defaults

Comparison jobs must be deterministic even before engine-specific controls are expanded.

| Engine ID | Display Name | MVP default mode | MVP default voice/lang | Clone behavior | Streaming behavior |
|---|---|---|---|---|---|
| `voxcpm` | VoxCPM2 | Voice design or clone when requested; plain TTS otherwise | Free-form voice description when supplied | Supported with `reference_wav_path`; no transcript required by current app flow | Existing single-engine streaming remains available outside comparison jobs |
| `supertonic` | Supertonic | Plain/preset TTS | Voice `M1`, language `en` unless user selects a supported preset later | Unsupported; fallback to plain/preset TTS with note | Unsupported in MVP |
| `neutts-nano` | NeuTTS Nano | Plain TTS if no reference is supplied; clone when reference audio and text are supplied | Default speaker/reference behavior is adapter-defined for plain TTS and must be reported in result metadata | Supported only when both `reference_wav_path` and `reference_text` are present; otherwise fail that engine result | Disabled in MVP comparison jobs |
| `kokoro` | Kokoro | Plain/preset TTS | `lang_code='a'`, voice `af_heart` unless user selects a supported preset later | Unsupported; fallback to plain/preset TTS with note | Disabled in MVP comparison jobs; pseudo-streaming is out of MVP scope |

Each result must include the effective voice/preset/lang metadata used by the adapter where applicable.

### Capability Validation Rules

| Request condition | Engine capability | MVP behavior |
|---|---|---|
| Voice description supplied | Engine supports voice design | Use voice description. |
| Voice description supplied | Engine does not support voice design | Generate plain supported TTS and add a fallback note. |
| Reference audio supplied for cloning | Engine does not support cloning | Generate plain supported TTS and add a fallback note. |
| Reference audio supplied for cloning | Engine supports cloning and does not require transcript | Clone using reference audio. |
| Reference audio supplied for cloning | Engine supports cloning and requires transcript, but `reference_text` is missing | Mark that engine result `failed`; do not generate fallback audio. |
| Reference audio + reference text supplied | Engine supports cloning and requires transcript | Clone using both inputs. |
| Selected engine is not installed | Engine unavailable | UI disables selection; API rejects submitted jobs with validation error. |

### Evaluation Strategy

Because this is an audio generation comparison tool, correctness is primarily workflow and traceability rather than subjective audio quality.

MVP evaluation must verify:

- Same input text is sent to every selected engine in a job.
- Unsupported voice-design/cloning inputs produce explicit fallback notes.
- Generated files are named with matching job/model identifiers.
- Each completed output is playable from the web UI.
- A failed generation from one installed engine does not prevent successful installed engines from producing outputs.
- CLI capability mismatches fail before model load where possible.

Subjective audio quality comparison remains user-driven and out of scope for automated evaluation.

---

## 4. Technical Specifications

### Architecture Overview

The update should introduce a shared multi-engine layer while keeping `core.py` as the source of truth.

Recommended components:

- `EngineCapabilities`: describes what each engine supports.
- `EngineStatus`: reports install/load/error state without forcing model load.
- `EngineRequest`: normalized request shape for one engine run.
- `EngineResult`: normalized result shape for completed, skipped, fallback, or failed outputs.
- `BaseTTSEngine` protocol/adapter interface.
- Engine registry keyed by stable `engine_id`.
- `ComparisonJob` request/response shape for web/API multi-engine jobs.
- Per-engine model state keyed by `engine_id`, replacing the current single global model assumption for web comparison jobs.

### Runtime and Load Lifecycle

MVP policy:

- Engines lazy-load on first use.
- Comparison generation runs one engine at a time.
- Only one engine may actively generate at a time.
- Loaded engines may remain resident after use for faster subsequent jobs unless memory pressure or adapter constraints require unloading.
- The top bar must distinguish `ready` from `currently_generating`.
- If a future implementation adds explicit unloading, it must update engine status immediately and must not change the API contract.

### Data Flow

```text
Web form
  -> POST /api/jobs
  -> create sanitized job_name
  -> resolve selected engines
  -> for each engine sequentially:
       - inspect capabilities
       - downgrade unsupported voice design/cloning to plain TTS with note
       - load engine if needed
       - generate audio
       - save as <job-name>-<engine-id>-<timestamp>.wav
       - append result to job response/history
  -> UI groups results by job_name
```

### API Requirements

#### Engine discovery

The API must expose an engine discovery endpoint:

```text
GET /api/engines
```

Response must include, per engine:

- `engine_id`
- `display_name`
- `installed`
- `status`
- `capabilities`
- `install_hint` when unavailable
- `load_error` when applicable

Unavailable engines:

- Must remain visible in the UI with `status=not_installed` and an install hint.
- Must be disabled for selection in the UI.
- Must be excluded from “select all”.
- Must cause API validation failure if submitted anyway.

#### Model status

The API must expose model load status without triggering model load, either through `/api/engines` or a dedicated endpoint.

#### Comparison generation

The API must expose a web-focused comparison endpoint:

```text
POST /api/jobs
```

Request fields:

- `job_name`: optional string; auto-generated if omitted
- `engine_ids`: non-empty list of selected engines
- `text`: required string
- `voice_description`: optional string
- `reference_wav_path`: optional string
- `reference_text`: optional string, required only for engines that support cloning and require transcript
- common generation parameters already supported by the app where applicable

Response fields:

- `job_name`
- `created_at`
- `results`: ordered list of per-engine `EngineResult`

Each `EngineResult` must include:

- `engine_id`
- `status`: `completed`, `completed_with_fallback`, `failed`, or `skipped`
- `capability_notes`: list of strings
- `file` and `url` when audio is produced
- `sample_rate` when known
- `duration_s` when audio is produced
- `elapsed_s`
- `error` when failed

#### Job history

The API must expose grouped job history:

```text
GET /api/jobs
GET /api/jobs/{job_name}
```

`GET /api/jobs` returns job summaries sorted newest first:

- `job_name`
- `created_at`
- `engine_ids`
- `status`
- `result_count`
- `completed_count`
- `failed_count`

`GET /api/jobs/{job_name}` returns the full job metadata and ordered per-engine results. The web UI must use these endpoints for grouped history after page refresh instead of reconstructing jobs from filenames.

### Web UI Requirements

- Add an engine selector that supports one, many, and all engines.
- Add an editable auto-generated job name field.
- Show top-bar engine status for all engines.
- Show active model load/generation progress during sequential generation.
- Group outputs by job name.
- Disable streaming controls whenever more than one engine is selected.
- Keep existing streaming as a single-engine action for engines whose capabilities report streaming support.
- Show each engine result under the group with:
  - engine display name
  - status
  - capability/fallback note
  - generated audio player when available
  - filename
  - duration and elapsed time when available
- Preserve current single-generation experience as the default path when only VoxCPM is selected.

### CLI Requirements

- CLI remains single-engine only.
- Add runtime engine selection with `--engine`.
- Add `--reference-text` for engines that require a reference transcript.
- Validate capability flags before generation.
- Error on incompatible options instead of silently falling back.
- CLI output naming may use the same `<job-name>-<engine-id>-<timestamp>.wav` convention when a job/name option is provided, but CLI comparison grouping is out of scope.

### Filename and Metadata Rules

Filename pattern:

```text
<job-name>-<engine-id>-<timestamp>.wav
```

Rules:

- `job-name` must be lowercase slug-compatible by default.
- User-edited names must be sanitized before use.
- `engine-id` must be stable and filesystem-safe, for example `voxcpm`, `supertonic`, `neutts-nano`, `kokoro`.
- Timestamp must be precise enough to avoid collisions during sequential generation.
- Job-based filenames must be constructed once from normalized job metadata. Collision fallback must preserve the visible `<job-name>-<engine-id>-<timestamp>` prefix and may append a numeric suffix before `.wav`.

MVP metadata persistence:

- Store job/result metadata in JSON sidecars under a lightweight `jobs/` directory or equivalent gitignored local metadata directory.
- Do not rely only on filenames for grouping.
- Metadata files must not contain audio payloads.

### Security & Privacy

- Preserve local-only server binding to `127.0.0.1`.
- Continue validating uploads by extension and size.
- Do not expose generated outputs outside local static serving.
- Do not send text, references, or generated audio to cloud services unless a future engine explicitly requires it and the UI labels that behavior.
- Missing optional dependencies must not expose stack traces in UI responses.

---

## 5. Risks & Roadmap

### Phased Rollout

#### MVP

- Engine registry and capabilities.
- Web/API multi-engine comparison jobs through `POST /api/jobs`.
- Grouped job history through `GET /api/jobs` and `GET /api/jobs/{job_name}`.
- Sequential generation.
- Grouped UI results.
- Filename convention with job + engine + timestamp.
- Top-bar model load status.
- CLI single-engine runtime selection with capability validation.
- Final-WAV-only comparison jobs; streaming remains single-engine.

#### v1.1

- Add richer job history filtering, sorting, and cleanup controls on top of MVP metadata sidecars.
- Add richer engine-specific controls, such as Kokoro `lang_code`, Supertonic voice preset/style JSON, and improved NeuTTS reference transcript UX.
- Add better first-load progress messaging where engine libraries expose download progress.

#### v2.0

- Optional controlled concurrency.
- Audio notes, ratings, and comparison annotations.
- Export job bundle containing WAVs and metadata.
- Optional streaming support per engine where reliable.

### Technical Risks

- Optional engines have different dependency stacks and may not install cleanly on every platform.
- Model first-load/download behavior may not expose granular progress events.
- Sequential generation avoids memory pressure but can make all-model comparison slow.
- Capability fallback can confuse users if notes are not visually prominent.
- Different engines produce different sample rates, which affects playback and comparison expectations.
- Keeping several engines resident after load may increase memory pressure; the sequential generation lock mitigates concurrent pressure but not total resident memory.

### Resolved MVP Decisions

- API route names are `GET /api/engines`, `POST /api/jobs`, `GET /api/jobs`, and `GET /api/jobs/{job_name}`.
- Job metadata persists in local JSON sidecars or equivalent gitignored local metadata files.
- Comparison jobs are final-WAV only.
- Existing streaming remains single-engine and capability-gated.
