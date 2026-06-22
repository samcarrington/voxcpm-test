# ADR-001: Streaming audio playback architecture for web UI

## Status
Accepted

## Date
2026-06-22

## Context
The streaming endpoint (`/api/generate/stream`) sends incremental audio chunks over WebSocket directly from the diffusion model's `generate_streaming()` loop. The browser-side `StreamPlayer` (in `web/app.js`) must play these chunks live — not after the generation completes — to give the user perceptual latency of seconds rather than tens of seconds.

Current implementation schedules each chunk immediately as it arrives:

```js
source.start(now);
this.nextStartTime = now + buffer.duration;
```

This "play-as-soon-as-it-arrives" approach produces choppy, stuttering audio because:

1. **Zero lookahead buffer** — scheduling points are anchored to the *current* time. Any gap between chunk arrivals (diffusion iteration variance, WebSocket frame delay) becomes an audible gap in playback.
2. **No cross-fade at boundaries** — each `AudioBufferSourceNode` is a hard cut. Chunk boundaries do not align with sample-perfect points, creating micro-clicks.
3. **No buffer drain awareness** — the player doesn't know how much audio is queued versus consumed. If generation outpaces playback, the schedule drifts. If playback outpaces generation, the buffer empties and the user hears silence.
4. **WebSocket delivery is unreliable** — frames are sent back-to-back (`send_bytes` + `send_json` per chunk). Browser event loop jitter, GC pauses, or network latency can cause variable inter-chunk delays.

The `streaming_adapter` in `core.py` already runs diffusion in a background thread and feeds chunks into an `asyncio.Queue` — so the server side can produce chunks independently of the WS send loop. The bottleneck is purely on the client scheduling layer.

## Decision

Replace the current `StreamPlayer` with a **ring-buffer scheduled player** that decouples generation speed from playback speed.

### Core design

- A contiguous `AudioBuffer` acts as a circular ring with read and write pointers.
- Chunks are **written** into the ring buffer as they arrive (circular wrap-around).
- A **drain ticker** (50ms interval) runs an infinite loop: reads from the ring buffer's read pointer, creates an `AudioBufferSourceNode` for the segment, cross-fades the transition region into the previously-playing source, and schedules it using `AudioContext.currentTime` plus a fixed latency offset (scheduling horizon).
- A **low-watermark rebuffer guard**: when the ring buffer fill drops below 200 ms of audio, the scheduler pauses enqueuing and waits (busy-while) until the buffer fills back to 800 ms. This eliminates micro-stutters caused by individual slow chunks.
- Chunk boundaries are **rounded up to 4096-sample multiples** on the server side (or rounded to 512-sample blocks client-side) so that every scheduled segment is sample-aligned, minimizing boundary artifacts.
- A **final flush marker** ("drain" or "done" signal) empties the remaining ring buffer, plays any tail with a 10 ms fade-out, then closes the `AudioContext`.

### Server-side change (minimal)

Add a `chunk_samples_round` parameter (default 4096) to the streaming loop in `webapp.py`. Each chunk is padded to the nearest multiple of `chunk_samples_round` before `send_bytes`. This makes boundaries predictable and ensures ring buffer segments are always sample-aligned.

### File changes

| File | Change |
|------|--------|
| `web/app.js` | Replace `StreamPlayer` (~40 LOC) with `RingBufferPlayer` (~200 LOC) |
| `webapp.py` | Add `chunk_samples_round` param to streaming loop, pad chunks |
| `web/index.html` | No change |
| `core.py` | No change |

### Scheduling timeline

```
AudioContext.currentTime  ────────►
            ┌─── segment 0 (hard start)
            │    └──────┐
            │    cross-fade (10ms)
            │    ┌──────┐
scheduler ◄─┴────┘      └──► drain ticker fires every 50ms
            ▲
      ring buffer fill level:
      [████████████░░░░░░░░░░░░░░░░░░░░░░░░]
         read ptr            write ptr

Ring buffer fill:
  > 800ms  → drain ticker schedules all segments up to read ptr + 1s horizon
  200–800ms → schedule only what's available, no wait
  < 200ms  → busy-wait; pause scheduling until buffer refills
```

## Alternatives Considered

### Option B: Batch Collector + Cross-fade

Batch chunks for a minimum 100ms wall-time before scheduling; add 10-20 ms cross-fade between segments.

- **Rejected as primary approach** because it doesn't solve the fundamental problem: if diffusion stalls or network drops a frame, the player still stalls. The batch collector only reduces *frequency* of stutters, not severity. It also adds ~200ms of latency even in ideal conditions.
- Kept as a fallback if the ring buffer proves too complex or buggy for the project scope.

### Option C: Blob URL + `<audio>` element (no live streaming)

Collect all chunks on the client, build a WAV blob, assign via `URL.createObjectURL()`.

- **Rejected** because it eliminates live streaming entirely. The user sees "Streaming..." for 10-30 seconds, hears nothing, and gets no benefit over the non-streaming `/api/generate` endpoint.

### Option D: HLS/m3u8 streaming

Server cuts chunks into transport stream segments; client plays via HLS.js.

- **Rejected** as over-engineered. HLS adds manifest management, segment file I/O, and client library dependency (hls.js) to a local-only TTS app. Latency advantage is negligible compared to ring buffer Web Audio.

### Option E: MediaSource Extensions (MSE)

Feed raw audio into a `MediaSource` with a `MediaSourceBuffer`.

- **Rejected** as premature. MSE is designed for video/audio container formats (MP4, WebM). Feeding PCM into a WebM container adds serialization overhead and complexity. For raw PCM streaming, the Web Audio API + ring buffer is simpler and more direct.

## Consequences

### Positive

- Perception of smooth audio from first chunk onward, regardless of inference speed variance.
- Ring buffer absorbs both fast bursts (back-pressure) and slow gaps (drain continues playing buffered audio).
- Cross-fade boundaries eliminate clicks at chunk transition points.
- Low-watermark guard handles worst-case scenarios (one bad diffusion iteration, network hiccup) without audible stutter.
- The `AudioContext` scheduling horizon (2-5 seconds ahead) keeps the Web Audio API's internal scheduler happy and prevents drift.

### Negative

- Increased `StreamPlayer` complexity: ~200 LOC replacing ~40. Ring buffer index math, circular wrap, and GC edge cases require careful testing.
- Larger memory footprint: ring buffer holds up to ~2 seconds of 48 kHz 16-bit mono = 192 KB. Negligible for most devices.
- Server-side chunk padding adds ~0.5-2 ms of CPU time per chunk (padded segment is immediately dropped at the read boundary). Worth it for alignment guarantees.

### Neutral

- No changes to `core.py` or `app.py` are necessary — the change is isolated to web server + client.
- The streaming endpoint already runs diffusion in a background thread; the ring buffer's async-aware scheduler fits the existing architecture naturally.
