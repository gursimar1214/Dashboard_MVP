"""
observability.py -- Langfuse tracing, fully optional.

ON only when the `langfuse` package is installed AND both LANGFUSE_PUBLIC_KEY
and LANGFUSE_SECRET_KEY are set. Otherwise every helper is a silent no-op and
the chatbot behaves exactly as without it.

Tracing never slows an answer (the SDK batches and exports in a background
thread) and never raises: a Langfuse problem cannot break a user's question.

One trace per question, grouped by chat session:

  chat-request                 input = question, output = answer + SQL + sample rows
  |- sql-plan   (generation)   messages, reply, tokens in/out, load / prompt-read /
  |                            generate seconds, tok/s
  |- validate   (span)         SQL after canonicalising; red if rejected
  |- execute    (span)         row count
  `- (retry: second sql-plan / validate / execute, attempt=2)

Works with Langfuse Python SDK v3 AND v4:
  * session_id / tags are set with propagate_attributes() around the root
    observation (the v4 way; v4 removed update_current_trace). Older v3
    builds without it fall back to update_current_trace().
  * Tags must be known up front in v4, so they carry only static facts
    (model, mode). What happened per request (path, resolution, timings,
    typo fixes...) goes in the root observation's metadata and in scores.

Env vars
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
  LANGFUSE_HOST / LANGFUSE_BASE_URL   default https://cloud.langfuse.com
  CHATBOT_LANGFUSE_ROWS               result rows sent per trace (default 5; 0 = none)
  CHATBOT_LANGFUSE_FULL_PROMPT        1 = send the whole system prompt on every
                                      generation (default 0: a short fingerprint
                                      instead -- the prompt is static code, and
                                      sending ~9 KB per trace is waste)
"""
from __future__ import annotations

import hashlib
import logging
import os
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator, Optional

log = logging.getLogger("chatbot.observability")

_client: Any = None
_enabled = False
_ROWS_IN_TRACE = int(os.environ.get("CHATBOT_LANGFUSE_ROWS", "5"))
_FULL_PROMPT = os.environ.get("CHATBOT_LANGFUSE_FULL_PROMPT", "0") == "1"


def init() -> bool:
    """Create the Langfuse client once, at server startup. True if tracing is
    on. Safe to call repeatedly."""
    global _client, _enabled
    if _client is not None:
        return _enabled
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        log.info("Langfuse tracing off (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set).")
        _client = False
        return False
    # The v4 SDK reads LANGFUSE_BASE_URL, v3 reads LANGFUSE_HOST: accept either.
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    if host:
        os.environ.setdefault("LANGFUSE_HOST", host)
        os.environ.setdefault("LANGFUSE_BASE_URL", host)
    try:
        from langfuse import get_client   # type: ignore
        client = get_client()
        if not client.auth_check():       # fail fast at startup, not on question 1
            raise RuntimeError("auth_check() returned False -- check the keys and the host/region")
        _client = client
        _enabled = True
        log.info("Langfuse tracing ON (host %s).", host or "https://cloud.langfuse.com")
    except Exception as exc:              # not installed, bad keys, host unreachable...
        log.warning("Langfuse tracing off (%s).", exc)
        _client = False
        _enabled = False
    return _enabled


def enabled() -> bool:
    return _enabled


def shutdown() -> None:
    """Send anything still queued. Call once on server stop."""
    if _enabled:
        for fn in ("flush", "shutdown"):
            try:
                getattr(_client, fn)()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Handle returned by the context managers. update() is always safe to call,
# on or off, so call sites never need an `if obs.enabled()` guard.
# --------------------------------------------------------------------------

class _Obs:
    __slots__ = ("_inner", "trace_id")

    def __init__(self, inner: Any = None, trace_id: Optional[str] = None):
        self._inner = inner
        self.trace_id = trace_id

    def update(self, **kwargs: Any) -> None:
        if self._inner is None:
            return
        try:
            self._inner.update(**kwargs)
        except Exception:
            log.debug("langfuse update failed", exc_info=True)


_NULL = _Obs()


@contextmanager
def observation(
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: Optional[dict] = None,
    **kwargs: Any,
) -> Iterator[_Obs]:
    """A span / generation nested under whatever observation is active."""
    if not _enabled:
        yield _NULL
        return
    try:
        cm = _client.start_as_current_observation(
            as_type=as_type, name=name, input=input, metadata=metadata, **kwargs
        )
        inner = cm.__enter__()
    except Exception:
        log.debug("langfuse start failed", exc_info=True)
        yield _NULL
        return
    try:
        trace_id = _client.get_current_trace_id()
    except Exception:
        trace_id = None
    try:
        yield _Obs(inner, trace_id)
    except BaseException as exc:
        try:
            inner.update(level="ERROR", status_message=str(exc)[:300])
        except Exception:
            pass
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            pass
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


@contextmanager
def request_trace(
    name: str,
    session_id: Optional[str],
    input: Any,
    tags: Optional[list[str]] = None,
) -> Iterator[_Obs]:
    """Root observation of one chat request, tagged with the chat session so a
    whole conversation replays together in Langfuse's Sessions view.

    The setup lives OUTSIDE the `yield`: an exception raised by the caller's
    code travels straight through (marking the trace as an error) instead of
    being caught here and followed by a second yield, which would crash the
    context manager and hide the real error."""
    if not _enabled:
        yield _NULL
        return
    with ExitStack() as stack:
        root = _NULL
        try:
            attrs: dict[str, Any] = {}
            if session_id:
                attrs["session_id"] = str(session_id)[:200]
            if tags:
                attrs["tags"] = [str(t)[:200] for t in tags]
            has_propagate = False
            try:
                from langfuse import propagate_attributes   # type: ignore
                has_propagate = True
            except ImportError:
                pass
            if has_propagate and attrs:
                stack.enter_context(propagate_attributes(**attrs))
            root = stack.enter_context(observation(name, as_type="span", input=input))
            if not has_propagate and attrs and root._inner is not None:
                try:
                    _client.update_current_trace(**attrs)     # older v3 builds
                except Exception:
                    log.debug("langfuse update_current_trace failed", exc_info=True)
        except Exception:
            log.debug("langfuse request_trace setup failed", exc_info=True)
        yield root


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def update_current_span(**kwargs: Any) -> None:
    if not _enabled:
        return
    try:
        _client.update_current_span(**kwargs)
    except Exception:
        log.debug("langfuse update_current_span failed", exc_info=True)


def prepare_messages(messages: list[dict]) -> list[dict]:
    """What to record as a generation's input. The system prompt is ~9 KB of
    static code; sending it with every trace is waste, so by default it is
    replaced by its length and a hash (a changed hash = a changed prompt).
    CHATBOT_LANGFUSE_FULL_PROMPT=1 sends it in full."""
    if _FULL_PROMPT:
        return messages
    out = []
    for m in messages:
        if m.get("role") == "system":
            body = str(m.get("content", ""))
            sha = hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]
            out.append({"role": "system",
                        "content": f"[system prompt: {len(body)} chars, sha1 {sha}; "
                                   "set CHATBOT_LANGFUSE_FULL_PROMPT=1 to log it in full]"})
        else:
            out.append(m)
    return out


def trim_rows(rows: Optional[list]) -> Any:
    """Result rows for a trace: the row count plus a few sample rows, so a
    200-row answer doesn't bloat (or leak into) Langfuse."""
    if rows is None:
        return None
    return {"row_count": len(rows), "sample_rows": rows[:_ROWS_IN_TRACE] if _ROWS_IN_TRACE > 0 else []}


def score(trace_id: Optional[str], name: str, value: Any, comment: Optional[str] = None) -> bool:
    """Attach a score to a trace; returns whether it was sent. Numbers become
    NUMERIC scores (thumbs, answered_from_data); strings become CATEGORICAL
    (path). The score id is (trace, name), so sending the same score again
    UPDATES it -- a changed thumbs-up/down leaves exactly one score."""
    if not (_enabled and trace_id):
        return False
    try:
        if isinstance(value, str):
            kwargs = {"value": value, "data_type": "CATEGORICAL"}
        else:
            kwargs = {"value": float(value), "data_type": "NUMERIC"}
        _client.create_score(trace_id=trace_id, name=name, comment=comment,
                             score_id=f"{trace_id}-{name}", **kwargs)
        return True
    except Exception:
        log.debug("langfuse score failed", exc_info=True)
        return False
