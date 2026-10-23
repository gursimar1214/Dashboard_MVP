"""
main.py -- stage 1.

Run with (from the project root, so `chatbot` is importable as a package):

    python -m uvicorn chatbot.main:app --host 127.0.0.1 --port 8000

Requires Ollama running locally with the model pulled first:

    ollama pull qwen2.5-coder:7b
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from . import llm, memory
from . import feedback, observability
from .config import FAST_MODE
from .pipeline import answer_question, warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("chatbot.main")


def _warm_in_background() -> None:
    """Load the model + prime the prompt cache without delaying server start,
    then log how much of the model landed on the GPU."""
    try:
        warmup()
    except Exception as exc:
        log.warning("Model warm-up skipped (%s).", exc)
        return
    status = llm.model_status()
    gpu = status.get("gpu_percent")
    if gpu is None:
        log.info("Model warmed up (GPU share unknown).")
    elif gpu == 0:
        log.warning(
            "Model warmed up but running on CPU ONLY. Run `python -m chatbot.check_model` for GPU help."
        )
    else:
        log.info("Model warmed up: %s%% on GPU.", gpu)


@asynccontextmanager
async def lifespan(app: FastAPI):
    observability.init()           # Langfuse tracing: on only if both keys are set
    threading.Thread(target=_warm_in_background, name="llm-warmup", daemon=True).start()
    yield
    observability.shutdown()       # flush any traces still queued


app = FastAPI(title="Dashboard Chatbot", version="1.7.0", lifespan=lifespan)


_SESSION_ID_RE = re.compile(r"^[\w.:-]{1,128}$")


class ChatRequest(BaseModel):
    # 2000 characters is ~500 tokens: plenty for a question, and it keeps one
    # message from crowding the schema prompt out of the model's context.
    message: str = Field(..., max_length=2000)
    session_id: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def _valid_session_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None          # treated as "no session yet": a new one is issued
        if not _SESSION_ID_RE.match(value):
            raise ValueError("session_id must be 1-128 characters: letters, digits, _ . : -")
        return value


class ChatResponse(BaseModel):
    answer: str
    sql: Optional[str] = None
    rows: Optional[list[dict[str, Any]]] = None
    intent: str
    # How a follow-up was understood ("revenue in india in 2022" for "what
    # about india"). None when the question was taken as typed.
    interpreted_as: Optional[str] = None
    session_id: str
    request_id: str
    answer_id: Optional[str] = None   # send this back with /feedback


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


@app.get("/health")
def health() -> dict:
    status = llm.model_status()
    return {
        "status": "ok",
        "fast_mode": FAST_MODE,
        "langfuse": observability.enabled(),
        "model": status["model"],
        "ollama_reachable": status["reachable"],
        "model_available": bool(status["reachable"] and status["installed"]),
        "model_loaded": status["loaded"],
        "gpu_percent": status["gpu_percent"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    request_id = request.state.request_id
    session_id = req.session_id or memory.new_session_id()
    started = time.perf_counter()
    try:
        result = answer_question(message, session_id=session_id)
    except Exception as exc:
        log.exception("request=%s failed", request_id)
        raise HTTPException(status_code=500, detail="internal error while answering") from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "request=%s session=%s intent=%s elapsed_ms=%d",
        request_id, session_id, result.intent, elapsed_ms,
    )
    return ChatResponse(**result.to_dict(), session_id=session_id, request_id=request_id)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Server-Sent Events variant: emits each pipeline stage as it happens,
    then the final payload as a `final` event. Clients that don't want SSE
    can keep using POST /chat."""
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")
    session_id = req.session_id or memory.new_session_id()

    def event_stream():
        events: "queue.Queue[tuple[str, dict] | None]" = queue.Queue()

        def worker() -> None:
            try:
                result = answer_question(
                    message,
                    session_id=session_id,
                    progress=lambda stage: events.put(("stage", {"stage": stage})),
                )
                payload = result.to_dict()
                payload["session_id"] = session_id
                events.put(("final", payload))
            except Exception as exc:
                log.exception("stream pipeline failed")
                events.put(("error", {"error": str(exc)}))
            finally:
                events.put(None)

        threading.Thread(target=worker, name="chat-stream", daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            name, data = item
            yield f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class FeedbackRequest(BaseModel):
    answer_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[0-9a-fA-F]+$")
    helpful: bool
    comment: Optional[str] = Field(default=None, max_length=500)


@app.post("/feedback")
def post_feedback(req: FeedbackRequest) -> dict:
    """Thumbs up/down (+ optional note) on an answer. Always written to the
    local feedback log (chatbot/feedback_log.jsonl) with the question and SQL
    that produced the answer; mirrored to Langfuse as the `user_feedback`
    score (1 = helpful, 0 = not) when tracing is on."""
    result = feedback.save_feedback(req.answer_id, req.helpful, req.comment)
    return {"status": "recorded" if (result["logged"] or result["langfuse"]) else "ignored", **result}


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    """Forget a session's conversation history."""
    memory.clear(req.session_id)
    return {"status": "cleared"}


@app.get("/session/{session_id}")
def session_snapshot(session_id: str) -> dict:
    """Introspection: what the chatbot remembers for this session."""
    return {"session_id": session_id, "turns": memory.snapshot(session_id)}


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response
