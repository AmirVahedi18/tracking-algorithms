from __future__ import annotations

import asyncio
import base64
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .detector import Detector
from .pipeline import TrackingPipeline
from .trackers import available

app = FastAPI(title="Tracking Algorithms Demo")

_STATE: dict = {"detector": None, "encoder": None}
_SENTINEL = object()


def get_detector() -> Detector:
    if _STATE["detector"] is None:
        _STATE["detector"] = Detector()
    return _STATE["detector"]


def get_encoder():
    if _STATE["encoder"] is None:
        from .reid import ReIDEncoder
        _STATE["encoder"] = ReIDEncoder()
    return _STATE["encoder"]


@app.get("/api/trackers")
def list_trackers():
    return {"trackers": available(), "device": config.resolve_device()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    vid = f"{uuid.uuid4().hex}{ext}"
    dest = config.UPLOAD_DIR / vid
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"file": vid, "name": file.filename}


@app.post("/api/first-frame")
async def first_frame(payload: dict):
    src = config.UPLOAD_DIR / Path(payload.get("file") or "").name
    if not src.exists():
        return JSONResponse({"error": "video not found"}, status_code=404)
    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return JSONResponse({"error": "cannot read frame"}, status_code=400)

    detector = get_detector()
    dets = detector.detect(frame, conf=config.FIRSTFRAME_CONF)
    names = detector.names
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    h, w = frame.shape[:2]
    return {
        "image": image, "width": w, "height": h,
        "detections": [
            {"box": [round(float(v), 1) for v in d.xyxy], "cls": d.cls,
             "name": names.get(d.cls, str(d.cls)), "score": round(d.score, 3)}
            for d in dets
        ],
    }


@app.get("/api/result/{name}")
def result(name: str):
    path = config.RESULT_DIR / Path(name).name
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.websocket("/ws/track")
async def ws_track(ws: WebSocket):
    await ws.accept()
    params = await ws.receive_json()
    file = params.get("file")
    tracker_key = params.get("tracker")
    selection = params.get("selection") or {}
    src = config.UPLOAD_DIR / Path(file or "").name
    if not file or not src.exists():
        await ws.send_json({"type": "error", "message": "video not found"})
        await ws.close()
        return

    job = uuid.uuid4().hex[:8]
    result_name = f"{Path(file).stem}_{tracker_key}_{job}.mp4"
    mot_name = f"{Path(file).stem}_{tracker_key}_{job}.txt"

    encoder = None
    try:
        detector = get_detector()
        needs_reid = any(t["key"] == tracker_key and t["needs_reid"] for t in available())
        if needs_reid:
            await ws.send_json({"type": "status", "message": "loading ReID model..."})
            encoder = get_encoder()
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"type": "error", "message": f"init failed: {e}"})
        await ws.close()
        return

    pipeline = TrackingPipeline(tracker_key, detector, encoder=encoder)
    gen = pipeline.run(str(src), str(config.RESULT_DIR / result_name),
                       str(config.RESULT_DIR / mot_name), selection=selection)
    loop = asyncio.get_event_loop()

    await ws.send_json({"type": "start", "result": result_name})
    try:
        while True:
            payload = await loop.run_in_executor(None, next, gen, _SENTINEL)
            if payload is _SENTINEL:
                break
            img = base64.b64encode(payload["jpeg"]).decode()
            await ws.send_json({
                "type": "frame",
                "image": f"data:image/jpeg;base64,{img}",
                "stats": payload["stats"],
                "detections": payload["detections"],
                "tracks": payload["tracks"],
                "frame": payload["frame"],
                "total": payload["total"],
            })
        await ws.send_json({"type": "done", "result": result_name})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        gen.close()
        try:
            await ws.close()
        except Exception:
            pass


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
          name="static")
