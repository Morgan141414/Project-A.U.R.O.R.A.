"""Local A.U.R.O.R.A. application: FastAPI backend for chat and topic learning."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import threading
import time
import webbrowser

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models.tokenizer import CharacterTokenizer
from models.transformer import DecoderOnlyTransformer, TransformerConfig
from training.topic_trainer import train_topic
from utils.hardware import get_device, system_vitals

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "ui"
app = FastAPI(title="A.U.R.O.R.A. Local Neural Studio")
app.mount("/ui", StaticFiles(directory=STATIC), name="ui")
device = get_device()
event_log = deque(maxlen=80)
training_state = {"active": False, "topic": None, "progress": "IDLE", "last_checkpoint": None}
model_lock = threading.Lock()
training_guard = threading.Lock()
model_cache = None


def log(message: str) -> None:
    event_log.appendleft({"time": time.strftime("%H:%M:%S"), "message": message})


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class TrainRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=160)
    materials: str = Field(min_length=100, max_length=200_000)
    epochs: int = Field(default=1, ge=1, le=10)


def load_model():
    global model_cache
    checkpoint = ROOT / "checkpoints" / "best.pt"
    if not checkpoint.exists():
        return None
    with model_lock:
        if model_cache is None:
            state = torch.load(checkpoint, map_location=device)
            config = TransformerConfig(**state["config"])
            model_cache = DecoderOnlyTransformer(config).to(device)
            model_cache.load_state_dict(state["model"])
            model_cache.eval()
    return model_cache


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    checkpoint = ROOT / "checkpoints" / "best.pt"
    return {"device": str(device).upper(), "model": "A.U.R.O.R.A. Local Transformer", "checkpoint": checkpoint.exists(), "training": training_state, "vitals": system_vitals()}


@app.get("/api/logs")
def logs():
    return {"logs": list(event_log)}


@app.post("/api/chat")
def chat(request: ChatRequest):
    if training_state["active"]:
        raise HTTPException(409, "Training is active; chat is paused to protect MPS memory")
    log(f'>> user_message("{request.message[:60]}")')
    model = load_model()
    if model is None:
        raise HTTPException(503, "No checkpoint found. Run a training session first.")
    tokenizer = CharacterTokenizer.load(ROOT / "data" / "tokenizer.json")
    prompt = request.message[-model.config.block_size + 1:]
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(ids, max_new_tokens=240, temperature=0.7)[0].tolist()
    answer = tokenizer.decode(generated[len(tokenizer.encode(prompt)):]).strip() or "I need more training data for this topic."
    log(">> local_inference // complete")
    return {"answer": answer, "device": str(device).upper(), "mode": "local-checkpoint"}


@app.post("/api/train")
def train(request: TrainRequest):
    if not training_guard.acquire(blocking=False):
        raise HTTPException(409, "A training run is already active")

    def worker():
        global model_cache
        training_state.update(active=True, topic=request.topic, progress="PREPARING")
        log(f">> start_learning // topic={request.topic}")
        try:
            training_state["progress"] = "TRAINING"
            checkpoint = train_topic(ROOT, request.topic, request.materials, request.epochs)
            training_state.update(progress="SUCCESS", last_checkpoint=str(checkpoint))
            with model_lock:
                model_cache = None
            log(f">> checkpoint_saved // {checkpoint.name}")
        except Exception as exc:
            training_state["progress"] = f"ERROR: {exc}"
            log(f">> training_error // {exc}")
        finally:
            training_state["active"] = False
            training_guard.release()

    threading.Thread(target=worker, daemon=True).start()
    return {"accepted": True, "message": "Learning session started"}


if __name__ == "__main__":
    import uvicorn
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
