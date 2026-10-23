"""
llm.py -- the only file that talks to the model.

Every call to the model goes through here so the rest of the pipeline works
with typed Python objects, never raw model text.

Structured calls (intent / SQL) are grammar-constrained through Ollama's
JSON-schema `format` parameter, which Ollama compiles to a GBNF grammar
internally. (The old optional Outlines backend was removed: it wrapped a
client with a 3-second probe timeout, so real generations timed out and
silently fell back to this same path -- slower, with no benefit.)

Concurrency: a semaphore caps in-flight calls at LLM_NUM_PARALLEL (default 1,
matching Ollama's own default and what an iGPU can handle).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Type, TypeVar

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from . import observability as obs
from .config import (
    LLM_KEEP_ALIVE,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_GPU,
    LLM_NUM_PARALLEL,
    LLM_NUM_THREAD,
    LLM_THINK,
    NL_MAX_TOKENS,
    HTTP_KEEPALIVE_S,
    OLLAMA_HOST,
    REQUEST_TIMEOUT_S,
)

log = logging.getLogger("chatbot.llm")

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------
# Errors -- each carries a message that is safe to show to the end user
# --------------------------------------------------------------------------

class LLMError(Exception):
    user_message = "The language model failed to answer -- please try again."


class LLMUnavailable(LLMError):
    user_message = (
        "I couldn't reach the language model -- check that Ollama is running "
        "(run `ollama serve`), then try again."
    )


class LLMModelMissing(LLMError):
    user_message = (
        f"The model '{LLM_MODEL}' isn't installed in Ollama -- run "
        f"`ollama pull {LLM_MODEL}` and try again."
    )


class LLMTimeout(LLMError):
    user_message = (
        "The language model took too long to answer (it may still be loading). "
        "Please try again in a moment."
    )


class LLMBadOutput(LLMError):
    user_message = (
        "The model's reply couldn't be understood -- try rephrasing the question "
        "a bit more simply."
    )


# --------------------------------------------------------------------------
# Client + request options
# --------------------------------------------------------------------------

_client: ollama.Client | None = None
_client_lock = threading.Lock()
_semaphore = threading.Semaphore(LLM_NUM_PARALLEL)

# Thinking models (Qwen3, ...) burn the token budget on hidden reasoning and
# can return empty JSON. Decide once whether to send think=False; if the
# server says the model has no thinking mode, stop sending it.
_send_think_false = (
    LLM_THINK == "off"
    or (LLM_THINK == "auto" and "qwen3" in LLM_MODEL.lower() and "coder" not in LLM_MODEL.lower())
)


def _get_client() -> ollama.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = ollama.Client(
                host=OLLAMA_HOST,
                timeout=REQUEST_TIMEOUT_S,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=HTTP_KEEPALIVE_S),
            )
        return _client


def _options(temperature: float, max_tokens: int) -> dict[str, Any]:
    """Options must be IDENTICAL on every call (except temperature/num_predict,
    which don't matter) -- if num_ctx/num_gpu change, Ollama reloads the model."""
    opts: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": LLM_NUM_CTX,
    }
    if LLM_NUM_GPU is not None:
        opts["num_gpu"] = LLM_NUM_GPU
    if LLM_NUM_THREAD is not None:
        opts["num_thread"] = LLM_NUM_THREAD
    return opts


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an ollama response object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _translate(exc: Exception) -> LLMError:
    if isinstance(exc, ollama.ResponseError):
        if getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower():
            return LLMModelMissing(str(exc))
        return LLMError(str(exc))
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeout(str(exc))
    if isinstance(exc, (ConnectionError, httpx.TransportError)):
        return LLMUnavailable(str(exc))
    return LLMError(str(exc))


def _log_stats(response: Any, wall: float | None = None) -> None:
    """One line per call: prompt tokens read, tokens written, and -- the part
    Ollama does not report -- `other`: wall time minus Ollama's own
    load/prompt/generate time. `other` should be ~0.1 s; a steady ~2 s means
    a (re)connection cost, not model time."""
    try:
        pe = _get(response, "prompt_eval_count") or 0
        ped = (_get(response, "prompt_eval_duration") or 0) / 1e9
        ev = _get(response, "eval_count") or 0
        evd = (_get(response, "eval_duration") or 0) / 1e9
        load = (_get(response, "load_duration") or 0) / 1e9
        other = f" | other={max(wall - (ped + evd + load), 0):.2f}s (wall {wall:.1f}s)" if wall is not None else ""
        log.info(
            "ollama: prompt_read=%d tok in %.1fs | generated=%d tok in %.1fs | load=%.1fs%s",
            pe, ped, ev, evd, load, other,
        )
    except Exception:
        pass


def _chat(messages: list[dict], options: dict, **kwargs: Any):
    """One call to Ollama's chat endpoint with shared settings."""
    global _send_think_false
    extra = dict(kwargs)
    if _send_think_false:
        extra["think"] = False
    try:
        t0 = time.perf_counter()
        response = _get_client().chat(
            model=LLM_MODEL,
            messages=messages,
            options=options,
            keep_alive=LLM_KEEP_ALIVE,
            **extra,
        )
        _log_stats(response, wall=time.perf_counter() - t0)
        return response
    except ollama.ResponseError as exc:
        if _send_think_false and "think" in str(exc).lower():
            log.info("Model has no thinking mode; no longer sending think=False.")
            _send_think_false = False
            return _chat(messages, options, **kwargs)
        raise


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _traced_chat(name: str, messages: list[dict], options: dict, **kwargs: Any):
    """_chat() recorded as a Langfuse *generation*: exact prompt, exact
    reply, token counts, and where the time went (model load / prompt
    reading / writing). A plain pass-through to _chat() when Langfuse is
    off -- this function costs nothing extra in that case."""
    params = {k: v for k, v in options.items() if isinstance(v, (int, float, str, bool))}
    traced_input = obs.prepare_messages(messages) if obs.enabled() else None
    with obs.observation(name, as_type="generation", input=traced_input,
                         model=LLM_MODEL, model_parameters=params) as gen:
        t_call = time.perf_counter()
        response = _chat(messages, options, **kwargs)
        wall_s = time.perf_counter() - t_call
        try:
            pe = _get(response, "prompt_eval_count") or 0
            ev = _get(response, "eval_count") or 0
            ped = (_get(response, "prompt_eval_duration") or 0) / 1e9
            evd = (_get(response, "eval_duration") or 0) / 1e9
            gen.update(
                output=(_get(_get(response, "message"), "content", "") or ""),
                usage_details={"input": pe, "output": ev},
                metadata={
                    "load_s": round((_get(response, "load_duration") or 0) / 1e9, 2),
                    "prompt_read_s": round(ped, 2),
                    "generate_s": round(evd, 2),
                    "other_s": round(max(wall_s - (ped + evd + (_get(response, "load_duration") or 0) / 1e9), 0), 2),
                    "prompt_tok_per_s": round(pe / ped, 1) if ped else None,
                    "output_tok_per_s": round(ev / evd, 1) if evd else None,
                    "done_reason": _get(response, "done_reason"),
                },
            )
        except Exception:
            pass
        return response


def generate_structured(system: str, prompt: str, schema: Type[T], max_tokens: int = 300, name: str = "sql-plan") -> T:
    """Grammar-constrained generation. Returns a validated `schema` instance,
    or raises an LLMError (with a user-safe message)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    json_schema = schema.model_json_schema()
    last_error: LLMError | None = None

    for attempt in (1, 2):
        # Second try gets more room in case the first reply was cut off.
        budget = max_tokens if attempt == 1 else int(max_tokens * 1.5)
        try:
            with _semaphore:
                response = _traced_chat(name, messages, _options(0, budget), format=json_schema)
            content = _get(_get(response, "message"), "content", "") or ""
            if _get(response, "done_reason") == "length":
                log.warning("Reply hit the %d-token limit (attempt %d).", budget, attempt)
            return schema.model_validate_json(content)
        except ValidationError as exc:
            last_error = LLMBadOutput(str(exc))
            log.warning("Unparsable JSON from model (attempt %d): %s", attempt, exc)
        except Exception as exc:
            err = _translate(exc)
            # Server down / model missing / timeout: a retry won't help.
            if isinstance(err, (LLMUnavailable, LLMModelMissing, LLMTimeout)):
                raise err from exc
            last_error = err
            log.warning("Ollama chat call failed (attempt %d): %s", attempt, exc)

        if attempt == 1:
            time.sleep(0.3)

    assert last_error is not None
    raise last_error


def generate_text(system: str, prompt: str, max_tokens: int = NL_MAX_TOKENS) -> str:
    """Unconstrained generation for stage 7 -- turning rows into prose."""
    try:
        with _semaphore:
            response = _traced_chat(
                "nl-reply",
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                _options(0.2, max_tokens),
            )
    except Exception as exc:
        raise _translate(exc) from exc
    return (_get(_get(response, "message"), "content", "") or "").strip()


# --------------------------------------------------------------------------
# Health / GPU status / warm-up
# --------------------------------------------------------------------------

def _normalise_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def model_status() -> dict[str, Any]:
    """Cheap health probe. Reports whether Ollama is up, whether the model is
    installed, and -- if it's loaded -- how much of it sits on the GPU."""
    status: dict[str, Any] = {
        "model": LLM_MODEL,
        "reachable": False,
        "installed": False,
        "loaded": False,
        "gpu_percent": None,
    }
    client = _get_client()
    try:
        listed = client.list()
    except Exception:
        return status
    status["reachable"] = True

    wanted = _normalise_tag(LLM_MODEL)
    installed = {
        _normalise_tag(_get(m, "model") or _get(m, "name") or "")
        for m in (_get(listed, "models", []) or [])
    }
    status["installed"] = wanted in installed

    try:
        running = client.ps()
    except Exception:
        return status
    for m in _get(running, "models", []) or []:
        if _normalise_tag(_get(m, "model") or _get(m, "name") or "") == wanted:
            size = _get(m, "size", 0) or 0
            vram = _get(m, "size_vram", 0) or 0
            status["loaded"] = True
            status["size_bytes"] = size
            status["vram_bytes"] = vram
            status["gpu_percent"] = round(100 * vram / size) if size else None
            break
    return status


def is_model_available() -> bool:
    s = model_status()
    return bool(s["reachable"] and s["installed"])


def warmup(system: str) -> None:
    """Load the model and pre-process the long system prompt once, so the
    first real question doesn't pay for it. Generates a single token; the
    prompt prefix stays in Ollama's cache for later calls."""
    try:
        with _semaphore:
            _chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Question: how many customers are there"},
                ],
                _options(0, 1),
            )
    except Exception as exc:
        raise _translate(exc) from exc
