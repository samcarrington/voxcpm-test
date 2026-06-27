import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any
from fastapi import HTTPException

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from core import (
    ALLOWED_UPLOAD_EXTS,
    ComparisonJobParams,
    EngineManager,
    GenerationParams,
    ModelState,
    OUTPUT_DIR,
    UPLOAD_DIR,
    ensure_output_dir,
    ensure_upload_dir,
    generate_comparison_job,
    generate_final,
    get_job,
    get_runtime_info,
    list_jobs,
    list_outputs,
    list_uploads,
    save_upload,
    assemble_prompt,
    apply_seed,
    build_generation_kwargs,
    normalize_engine_id,
    save_wav,
    STREAMING_STEPS,
    streaming_adapter,
)

DEFAULT_PORT = 8000
HOST = "127.0.0.1"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

logger = logging.getLogger("voxcpm.webapp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

model_state = ModelState()
engine_manager = EngineManager(model_state)
app = FastAPI(title="VoxCPM2 TTS", version="1.0")


@app.get("/api/status")
async def api_status(): return JSONResponse({"state": model_state.state})

@app.get("/api/info")
async def api_info(): return JSONResponse(await asyncio.to_thread(get_runtime_info))

@app.get("/api/engines")
async def api_engines(): return JSONResponse([e.model_dump() for e in engine_manager.list_engines()])

@app.post("/api/jobs")
async def api_jobs(params: ComparisonJobParams):
    try:
        job = await generate_comparison_job(engine_manager, params)
        return JSONResponse(job.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("job failed")
        raise HTTPException(status_code=500, detail="Job failed")

@app.get("/api/jobs")
async def api_jobs_list(): return JSONResponse([j.model_dump() for j in list_jobs()])

@app.get("/api/jobs/{job_name}")
async def api_jobs_detail(job_name: str):
    try:
        return JSONResponse(get_job(job_name).model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")

@app.post("/api/generate")
async def api_generate(params: GenerationParams):
    try:
        return JSONResponse(await generate_final(model_state, params, mode="ui_generate"))
    except Exception:
        logger.exception("generate failed")
        raise HTTPException(status_code=500, detail="Generation failed")

@app.post("/api/uploads")
async def api_upload_reference(reference: UploadFile = File(...)):
    data = await reference.read()
    if len(data) > MAX_UPLOAD_BYTES: return JSONResponse(status_code=400, content={"error": "File too large"})
    if not reference.filename: return JSONResponse(status_code=400, content={"error": "No filename provided"})
    if Path(reference.filename).suffix.lower() not in ALLOWED_UPLOAD_EXTS: return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    path = await asyncio.to_thread(save_upload, reference.filename, data)
    return JSONResponse({"path": str(path), "name": path.name})

@app.get("/api/uploads")
async def api_list_uploads(): return JSONResponse(await asyncio.to_thread(list_uploads))

@app.get("/api/outputs")
async def api_list_outputs(): return JSONResponse(await asyncio.to_thread(list_outputs))

@app.websocket("/api/generate/stream")
async def api_generate_stream(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_json()
        if msg.get("type") != "start":
            await ws.send_json({"type": "error", "message": "expected start"})
            return
        payload = msg.get("params", {})
        params = GenerationParams(**payload)
        engine_id = normalize_engine_id(payload.get("engine_id", msg.get("engine_id", "voxcpm")))
        engine = engine_manager.get(engine_id)
        if engine_id != "voxcpm" or not engine.capabilities.supports_streaming:
            await ws.send_json({"type": "error", "message": "Streaming is only available for VoxCPM"})
            await ws.close()
            return
        model = await model_state.get_or_load()
        apply_seed(params.seed)
        kwargs = build_generation_kwargs(
            text=assemble_prompt(params.text, params.voice_description),
            cfg_value=params.cfg_value,
            inference_timesteps=params.inference_timesteps if params.inference_timesteps != 20 else STREAMING_STEPS,
            normalize=params.normalize,
            min_len=2,
            max_len=4096,
            retry_badcase=False,
            retry_badcase_max_times=5,
            retry_badcase_ratio_threshold=5.0,
            reference_wav_path=params.reference_wav_path,
        )
        await ws.send_json({"type": "meta", "sample_rate": model.tts_model.sample_rate, "channels": 1})
        chunks: list[Any] = []
        t0 = time.perf_counter()
        async with model_state.lock:
            async for chunk in streaming_adapter(model, kwargs):
                chunks.append(chunk)
                arr = chunk if hasattr(chunk, "astype") else np.asarray(chunk)
                await ws.send_bytes(arr.astype("float32").tobytes())
                await ws.send_json({"type": "progress", "chunk_samples": int(arr.shape[-1])})
        if not chunks:
            raise RuntimeError("No audio produced")
        wav = np.concatenate(chunks)
        path = await asyncio.to_thread(save_wav, wav, "stream.wav", model.tts_model.sample_rate)
        await ws.send_json({"type": "saved", "url": f"/outputs/{path.name}", "file": path.name})
        await ws.send_json({"type": "done", "elapsed_s": time.perf_counter() - t0})
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("stream failed")
        try:
            await ws.send_json({"type": "error", "message": "Streaming failed"})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

ensure_output_dir(); ensure_upload_dir()
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.is_dir(): app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxCPM2 TTS web server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("webapp:app", host=HOST, port=args.port, reload=args.reload)
