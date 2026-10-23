"""
feedback.py -- the feedback loop. Works with OR without Langfuse.

  answer  -> every reply gets an answer_id (the Langfuse trace id when tracing
             is on, otherwise a random id). The server remembers the last
             few hundred answers (question, SQL, how it was understood...).
  thumbs  -> the dashboard sends POST /feedback {answer_id, helpful, comment}.
             main.py calls save_feedback(): one JSON line is appended to
             FEEDBACK_LOG (question + SQL are looked up from the remembered
             answer), and -- if tracing is on -- a `user_feedback` score is
             attached to the Langfuse trace.
  review  -> `python -m chatbot.feedback_review` lists the thumbs-down
             answers and drafts regression-test cases from them.

The log is append-only. If someone changes their mind, a second line is
written; the review tool uses the LAST line per answer_id.

Env: CHATBOT_FEEDBACK_LOG  path of the log (default: chatbot/feedback_log.jsonl)
     CHATBOT_FEEDBACK_KEEP how many recent answers to remember (default 500)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Optional

from . import observability

log = logging.getLogger("chatbot.feedback")

FEEDBACK_LOG = Path(os.environ.get("CHATBOT_FEEDBACK_LOG", Path(__file__).with_name("feedback_log.jsonl")))
_KEEP = int(os.environ.get("CHATBOT_FEEDBACK_KEEP", "500"))

_recent: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_lock = threading.Lock()


def _clip(text: Optional[str], n: int) -> Optional[str]:
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def remember_answer(answer_id: str, **fields: Any) -> None:
    """Called for every answer. Keeps the last _KEEP so a later thumbs click
    can be tied back to the question and SQL that produced it."""
    entry = {k: (_clip(v, 600) if isinstance(v, str) else v) for k, v in fields.items()}
    with _lock:
        _recent[answer_id] = entry
        _recent.move_to_end(answer_id)
        while len(_recent) > _KEEP:
            _recent.popitem(last=False)


def save_feedback(answer_id: str, helpful: bool, comment: Optional[str] = None) -> dict[str, Any]:
    """Log one thumbs up/down (+ optional note); mirror it to Langfuse."""
    with _lock:
        ctx = dict(_recent.get(answer_id, {}))
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "answer_id": answer_id,
        "helpful": bool(helpful),
        "comment": _clip(comment, 500),
        **ctx,
    }
    logged = False
    try:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _lock, FEEDBACK_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logged = True
    except (OSError, ValueError) as exc:      # unwritable / invalid path: never break a thumbs click
        log.warning("could not write feedback log %s: %s", FEEDBACK_LOG, exc)
    sent = observability.score(
        ctx.get("trace_id"), "user_feedback", 1.0 if helpful else 0.0, comment=entry["comment"],
    )
    return {"logged": logged, "langfuse": sent, "known_answer": bool(ctx)}


def read_log(path: Optional[Path] = None) -> Iterator[dict[str, Any]]:
    p = Path(path) if path else FEEDBACK_LOG
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def latest_per_answer(entries: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """The final verdict for each answer (a changed mind overrides)."""
    last: dict[str, dict[str, Any]] = {}
    for e in entries:
        last[e.get("answer_id", "")] = e
    return list(last.values())
