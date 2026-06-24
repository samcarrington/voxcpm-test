"""FastAPI web server for VoxCPM2 TTS. Local-only (binds 127.0.0.1).
Run with: python webapp.py [--port N]
"""

from contextlib import asynccontextmanager
import argparse
import asyncio
import logging
import time as _time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from core import (
    ModelState, OUTPUT_DIR, UPLOAD_DIR,
    list_outputs, list_uploads, save_upload, ALLOWED_UPLOAD_EXTS,
    save_wav, build_generation_kwargs, generate_with_retry,
    generate_final, streaming_adapter, get_runtime_info, show_info,
    detect_device, get_runtime_device, assemble_prompt, apply_seed,
    ensure_output_dir, ensure_upload_dir,
    DEFAULT_CFG, DEFAULT_STEPS, STREAMING_STEPS,
    GenerationParams,
)

DEFAULT_PORT = 8000
HOST = "127.0.0.1"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

logger = logging.getLogger("voxcpm.webapp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

model_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting")
    app.state.model_state = model_state
    yield
    logger.info("Server shutting down")


app = FastAPI(title="VoxCPM2 TTS", version="1.0", lifespan=lifespan)


@app.get("/api/status")
async def api_status():
    return JSONResponse({"state": model_state.state})


@app.get("/api/info")
async def api_info():
    info = await asyncio.to_thread(get_runtime_info)
    return JSONResponse(info)


@app.post("/api/generate")
async def api_generate(params: GenerationParams):
    try:
        result = await generate_final(model_state, params, mode="ui_generate")
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Generate failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/uploads")
async def api_upload_reference(reference: UploadFile = File(...)):
    try:
        data = await reference.read()
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=400,
                content={"error": f"File too large (max {MAX_UPLOAD_BYTES} bytes)"},
            )
        ext = Path(reference.filename).suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTS:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file type"},
            )
        path = await asyncio.to_thread(save_upload, reference.filename, data)
        return JSONResponse({"path": str(path), "name": path.name})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/uploads")
async def api_list_uploads():
    return JSONResponse(await asyncio.to_thread(list_uploads))


@app.get("/api/outputs")
async def api_list_outputs():
    return JSONResponse(await asyncio.to_thread(list_outputs))


@app.websocket("/api/generate/stream")
async def api_generate_stream(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_json()
        if msg.get("type") != "start":
            await ws.send_json({"type": "error", "message": "expected start"})
            await ws.close()
            return
        params = GenerationParams(**msg["params"])
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return

    try:
        apply_seed(params.seed)
        model = await model_state.get_or_load()

        timesteps = params.inference_timesteps if params.inference_timesteps != 20 else STREAMING_STEPS

        kwargs = build_generation_kwargs(
            text=assemble_prompt(params.text, params.voice_description),
            cfg_value=params.cfg_value,
            inference_timesteps=timesteps,
            normalize=params.normalize,
            min_len=2,
            max_len=4096,
            retry_badcase=False,
            retry_badcase_max_times=5,
            retry_badcase_ratio_threshold=5.0,
            reference_wav_path=params.reference_wav_path,
        )

        sample_rate = model.tts_model.sample_rate
        await ws.send_json({"type": "meta", "sample_rate": sample_rate, "channels": 1})

        t0 = _time.perf_counter()
        chunks: list[np.ndarray] = []
        model_state._init_locks()
        async with model_state.lock:
            async for chunk in streaming_adapter(model, kwargs):
                chunks.append(chunk)
                raw = chunk.astype(np.float32).tobytes()
                await ws.send_bytes(raw)
                await ws.send_json({
                    "type": "progress",
                    "chunk_samples": len(chunk),
                    "total_samples_so_far": sum(len(c) for c in chunks),
                })

        wav = np.concatenate(chunks)
        path = await asyncio.to_thread(save_wav, wav, "ui_stream.wav", sample_rate)
        elapsed = _time.perf_counter() - t0

        await ws.send_json({
            "type": "saved",
            "url": f"/outputs/{path.name}",
            "total_duration_s": len(wav) / sample_rate,
        })
        await ws.send_json({"type": "done", "elapsed_s": elapsed})
    except WebSocketDisconnect:
        return
    except Exception as e:
        logger.error(f"Stream failed: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        finally:
            await ws.close()


ensure_output_dir()
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:
    @app.get("/")
    def index():
        return JSONResponse({"info": "web/ directory not yet created"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxCPM2 TTS web server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(
        "webapp:app",
        host=HOST,
        port=args.port,
        reload=args.reload,
    )
