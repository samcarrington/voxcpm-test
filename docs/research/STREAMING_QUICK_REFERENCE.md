# VoxCPM Streaming - Quick Reference

## TL;DR

- **Streaming API:** `model.generate_streaming(text="...")`
- **Yields:** numpy float32 arrays, 1,280 samples each (80ms @ 16kHz)
- **Sample Rate:** 16,000 Hz (NOT 48kHz)
- **No built-in server:** You must wrap with FastAPI/WebSocket
- **Latency:** ~100-500ms per chunk (real-time factor: 0.5-2x)

---

## Quick Start

```python
from voxcpm import VoxCPM

# Load model (2-4 GB GPU memory)
model = VoxCPM.from_pretrained("openbmb/VoxCPM2")

# Stream audio chunks
for chunk in model.generate_streaming(
    text="Hello world",
    cfg_value=2.0,           # Guidance scale
    inference_timesteps=10   # Quality vs speed tradeoff
):
    # chunk: numpy.ndarray, shape (1280,), dtype float32
    # Range: [-1.0, 1.0]
    # Duration: 80ms
    print(f"Got chunk: {chunk.shape}")
```

---

## Chunk Details

| Property | Value |
|----------|-------|
| Shape | `(1280,)` |
| Dtype | `float32` |
| Sample Rate | 16,000 Hz |
| Duration | 80 milliseconds |
| Range | [-1.0, 1.0] |
| Encoding | PCM (linear) |

---

## Key Parameters

```python
model.generate_streaming(
    text="...",                    # Required: text to synthesize
    cfg_value=2.0,                 # Guidance (0.1-10.0, default 2.0)
    inference_timesteps=10,        # Steps (1-100, default 10)
    min_len=2,                     # Minimum audio length
    max_len=4096,                  # Maximum audio length
    streaming_prefix_len=3,        # Context patches (VoxCPM: 3, VoxCPM2: 4)
    
    # Optional:
    prompt_wav_path=None,          # For continuation mode
    prompt_text=None,              # Text for prompt audio
    reference_wav_path=None,       # Voice cloning (VoxCPM2 only)
    normalize=False,               # Text normalization
    denoise=False,                 # Denoise prompt/reference
)
```

---

## Converting Chunks to Audio

### Option 1: Raw PCM (int16)
```python
import numpy as np

for chunk in model.generate_streaming(text="..."):
    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    pcm_int16 = (chunk * 32767).astype(np.int16)
    # Send pcm_int16.tobytes() to client
```

### Option 2: WAV Format
```python
import soundfile as sf
import io

for chunk in model.generate_streaming(text="..."):
    buffer = io.BytesIO()
    sf.write(buffer, chunk, 16000, format='WAV')
    # Send buffer.getvalue() to client
```

### Option 3: Opus (Compressed)
```python
import opuslib
import numpy as np

encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_VOIP)

for chunk in model.generate_streaming(text="..."):
    pcm_int16 = (chunk * 32767).astype(np.int16)
    opus_bytes = encoder.encode(pcm_int16, 1280)
    # Send opus_bytes to client
```

---

## FastAPI Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from voxcpm import VoxCPM
import soundfile as sf
import io

app = FastAPI()
model = VoxCPM.from_pretrained("openbmb/VoxCPM2")

@app.post("/synthesize")
async def synthesize(text: str):
    def audio_generator():
        for chunk in model.generate_streaming(text=text):
            buffer = io.BytesIO()
            sf.write(buffer, chunk, 16000, format='WAV')
            yield buffer.getvalue()
    
    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav"
    )

# Run: uvicorn app:app --reload
```

---

## WebSocket Streaming (Low Latency)

```python
from fastapi import FastAPI, WebSocket
from voxcpm import VoxCPM
import numpy as np
import json

app = FastAPI()
model = VoxCPM.from_pretrained("openbmb/VoxCPM2")

@app.websocket("/ws/synthesize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Receive synthesis request
    data = await websocket.receive_json()
    text = data["text"]
    
    # Stream audio chunks
    for chunk in model.generate_streaming(text=text):
        # Convert to int16 PCM
        pcm_int16 = (chunk * 32767).astype(np.int16)
        
        # Send as binary message
        await websocket.send_bytes(pcm_int16.tobytes())
    
    await websocket.close()
```

---

## Performance Notes

### Latency
- **First chunk:** ~1-2 seconds (model warmup)
- **Per chunk:** ~100-500ms (depends on `inference_timesteps`)
- **Real-time factor:** 0.5-2x (generates 80ms in 100-500ms)

### Memory
- **Model:** 2-4 GB GPU VRAM
- **Peak:** 6-8 GB during inference
- **Batch size:** 1 (streaming doesn't batch)

### Quality vs Speed
- `inference_timesteps=4`: Fast, lower quality
- `inference_timesteps=10`: Balanced (default)
- `inference_timesteps=30`: Slow, high quality

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `inference_timesteps`
- Use smaller GPU or CPU (slower)
- Reduce batch size (already 1)

### "Audio sounds robotic/choppy"
- Increase `inference_timesteps` (10→20)
- Adjust `cfg_value` (try 1.5-3.0)
- Ensure `streaming_prefix_len` is not too small

### "Chunks are delayed"
- This is normal: inference takes 100-500ms per 80ms chunk
- Use WebSocket for lower perceived latency
- Consider buffering 2-3 chunks before playback

---

## Architecture Comparison

### HTTP Streaming (Simple)
```
Client → POST /synthesize → FastAPI → StreamingResponse → Client
```
- ✓ Simple to implement
- ✗ Higher latency perception
- ✗ No bidirectional communication

### WebSocket (Recommended)
```
Client ←→ WebSocket ←→ FastAPI ←→ VoxCPM
```
- ✓ Low latency
- ✓ Bidirectional (can cancel mid-stream)
- ✓ Better UX

### Server-Sent Events (SSE)
```
Client ← SSE ← FastAPI ← VoxCPM
```
- ✓ Simpler than WebSocket
- ✓ Works with HTTP/1.1
- ✗ Unidirectional only

---

## Files to Reference

- **Core API:** `voxcpm/core.py` (lines 157-158)
- **Streaming Logic:** `voxcpm/model/voxcpm.py` (lines 821-826)
- **Audio VAE:** `voxcpm/modules/audiovae/audio_vae.py` (lines 277, 333)
- **Full Report:** `VOXCPM_STREAMING_REPORT.md`

