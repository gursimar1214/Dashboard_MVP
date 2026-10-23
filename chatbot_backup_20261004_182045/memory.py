"""
memory.py -- short per-session conversation memory for follow-up questions.

Design
------
  - Each chat session has a session_id (main.py creates one on the first
    request; the client sends it back on every later request).
  - For each *answered* turn we keep: what the user typed, what it MEANT
    once made self-contained ("resolved question"), the SQL that ran, and a
    short description of the result. The resolved question is what makes a
    chain work: "and last month?" after "what about india?" after
    "revenue in germany" is stored as "revenue in india last month", so the
    third turn does not depend on decoding the second.
  - A turn whose SQL ran is remembered even when it returned no rows
    ("what about 2019?" after an empty answer must still attach to it).
    Turns that failed validation/execution, and out-of-scope turns, are not.
  - `prompt_context()` renders the history for the model and returns it
    together with its cache fingerprint, read under ONE lock so the two can
    never disagree.
  - The rendered history has a hard size budget, so it cannot push the
    schema prompt out of the model's context window.
  - Requests in the same session are serialised (`turn_gate`), so two
    quick messages are processed in order and the second sees the first.

Session lifetime: a session is forgotten after MEMORY_TTL_S of inactivity.
The TTL is enforced on every read (not just when something is written) and
any access by the user counts as activity.

Storage is an in-process dict; it does NOT survive a restart and does NOT
work across multiple uvicorn workers. Fine for one dashboard, one process.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Deque, Iterator, Optional

from .config import (
    MEMORY_CONTEXT_CHARS,
    MEMORY_MAX_SESSIONS,
    MEMORY_MAX_TURNS,
    MEMORY_TTL_S,
    MEMORY_TURN_WAIT_S,
)

MAX_TURNS_PER_SESSION = MEMORY_MAX_TURNS
MAX_SESSIONS = MEMORY_MAX_SESSIONS
SESSION_TTL_SECONDS = MEMORY_TTL_S
MAX_CONTEXT_CHARS = MEMORY_CONTEXT_CHARS

# The newest turns are shown in full (question, meaning, SQL, result); older
# ones only as "meaning -> result" to save room.
FULL_DETAIL_TURNS = 2

_MAX_QUESTION_CHARS = 300
_MAX_SUMMARY_CHARS = 400
_MAX_SQL_CHARS = 1600


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class _Turn:
    question: str           # exactly what the user typed (one line, capped)
    resolved: str           # the same question made self-contained
    sql: str                # SQL that ran, WITHOUT the injected row cap
    intent: str
    summary: str            # one-line description of the result
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    at: float = field(default_factory=time.time)
    # Structured slots of this turn (metric / grouping / filters / time), or
    # None when Python could not read the question completely. See slotmem.py.
    slots: Optional[dict] = None


@dataclass
class _Session:
    turns: Deque[_Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS_PER_SESSION))
    last_seen: float = field(default_factory=time.time)


@dataclass
class _Gate:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


_sessions: "OrderedDict[str, _Session]" = OrderedDict()
_gates: dict[str, _Gate] = {}
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def new_session_id() -> str:
    return uuid.uuid4().hex


def _one_line(text: Optional[str], limit: int) -> str:
    """Collapse all whitespace (incl. newlines) and cap the length. Stored
    text goes back into a prompt, so a multi-line question must not be able
    to fake extra 'Question:' / 'SQL:' lines in the history."""
    collapsed = re.sub(r"\s+", " ", (text or "")).strip()
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 3].rstrip() + "..."
    return collapsed


def _expired(session: _Session, now: float) -> bool:
    return now - session.last_seen > SESSION_TTL_SECONDS


def _get_locked(session_id: str, touch: bool = True) -> Optional[_Session]:
    """Fetch a live session (caller holds _lock). Expired sessions are
    deleted here, so an idle session can never leak stale context.
    touch=True counts the access as activity (refreshes TTL and LRU order);
    introspection passes touch=False."""
    session = _sessions.get(session_id)
    if session is None:
        return None
    now = time.time()
    if _expired(session, now):
        _sessions.pop(session_id, None)
        return None
    if touch:
        session.last_seen = now
        _sessions.move_to_end(session_id)
    return session


def _sweep_locked() -> None:
    """Drop stale sessions, then evict least-recently-used beyond the cap."""
    now = time.time()
    for sid in [sid for sid, s in _sessions.items() if _expired(s, now)]:
        _sessions.pop(sid, None)
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)


# --------------------------------------------------------------------------
# Rendering for the prompt
# --------------------------------------------------------------------------

def _render_turn(turn: _Turn, number: int, total: int, full: bool) -> str:
    newest = number == total
    if newest:
        label = f"Turn {number} (most recent)"
    else:
        label = f"Turn {number}"
    lines = [label]
    if full:
        lines.append(f"  User typed: {turn.question}")
        if turn.resolved and turn.resolved != turn.question:
            lines.append(f"  Meaning (self-contained): {turn.resolved}")
        lines.append(f"  SQL that ran: {turn.sql}")
        lines.append(f"  Result: {turn.summary}")
    else:
        lines.append(f"  Meaning: {turn.resolved or turn.question}")
        lines.append(f"  Result: {turn.summary}")
    return "\n".join(lines)


def _render(turns: list[_Turn]) -> str:
    """Oldest -> newest. The newest turn is always included in full. Older
    turns are added newest-first while they fit in MAX_CONTEXT_CHARS; a turn
    that does not fit is dropped whole, never cut in half (half a SQL
    statement is worse than none)."""
    total = len(turns)
    blocks: list[str] = []
    used = 0
    for idx in range(total - 1, -1, -1):          # newest first
        number = idx + 1
        full = (total - idx) <= FULL_DETAIL_TURNS
        block = _render_turn(turns[idx], number, total, full)
        cost = len(block) + 1
        if blocks and used + cost > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += cost
    blocks.reverse()
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def prompt_context(session_id: Optional[str]) -> tuple[str, str]:
    """(history text for the prompt, cache fingerprint) for this session,
    read atomically. Both are '' when there is nothing remembered."""
    if not session_id:
        return "", ""
    with _lock:
        session = _get_locked(session_id)
        if session is None or not session.turns:
            return "", ""
        turns = list(session.turns)
    fingerprint = "|".join(t.turn_id for t in turns)
    return _render(turns), fingerprint


def get_context(session_id: Optional[str]) -> str:
    """Rendered history for this session, or ''."""
    return prompt_context(session_id)[0]


def context_fingerprint(session_id: Optional[str]) -> str:
    """Stable id of the session's current history, for cache keys. Same
    words asked in two different conversations must not collide."""
    return prompt_context(session_id)[1]


def remember(
    session_id: Optional[str],
    question: str,
    sql: Optional[str],
    intent: str,
    summary: str = "",
    resolved_question: str = "",
    slots: Optional[dict] = None,
) -> None:
    """Record one answered turn. `sql` should be the query as the model
    wrote it (before the row cap was injected). `slots` is the turn's
    structured slots (slotmem.Slots.to_dict()) or None. No-op without a
    session or SQL."""
    if not session_id or not sql:
        return
    q = _one_line(question, _MAX_QUESTION_CHARS)
    turn_fields = dict(
        question=q,
        resolved=_one_line(resolved_question, _MAX_QUESTION_CHARS) or q,
        sql=_one_line(sql, _MAX_SQL_CHARS),
        intent=str(intent or ""),
        summary=_one_line(summary, _MAX_SUMMARY_CHARS),
    )
    with _lock:
        session = _get_locked(session_id, touch=False)
        if session is None:
            session = _Session()
            _sessions[session_id] = session
        last = session.turns[-1] if session.turns else None
        if last is not None and last.question == turn_fields["question"] and last.sql == turn_fields["sql"]:
            # Same question re-asked (or served from cache): refresh the
            # turn instead of filling the window with duplicates. turn_id
            # is kept, so the fingerprint -- and therefore the cache -- is
            # unchanged, which is correct: the history did not change.
            last.resolved = turn_fields["resolved"]
            last.summary = turn_fields["summary"]
            last.slots = slots
            last.at = time.time()
        else:
            session.turns.append(_Turn(**turn_fields, slots=slots))
        session.last_seen = time.time()
        _sessions.move_to_end(session_id)
        _sweep_locked()


def last_slots(session_id: Optional[str]) -> Optional[dict]:
    """Slots of the MOST RECENT turn, or None (no session, no turns, or the
    last turn could not be read into slots). Only the latest turn counts: a
    follow-up always attaches to what was just answered."""
    if not session_id:
        return None
    with _lock:
        session = _get_locked(session_id, touch=False)
        if not session or not session.turns:
            return None
        return session.turns[-1].slots


def clear(session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _lock:
        _sessions.pop(session_id, None)


def turn_count(session_id: Optional[str]) -> int:
    if not session_id:
        return 0
    with _lock:
        session = _get_locked(session_id)
        return len(session.turns) if session else 0


def snapshot(session_id: Optional[str]) -> list[dict]:
    """For the /session/{id} introspection endpoint. Read-only: looking does
    not keep a session alive."""
    if not session_id:
        return []
    with _lock:
        session = _get_locked(session_id, touch=False)
        if not session:
            return []
        return [
            {"question": t.question, "resolved_question": t.resolved, "sql": t.sql,
             "intent": t.intent, "summary": t.summary, "slots": t.slots, "at": t.at}
            for t in session.turns
        ]


@contextmanager
def turn_gate(session_id: Optional[str]) -> Iterator[None]:
    """Serialise the requests of ONE session (different sessions never wait
    for each other). Without this, a second message sent while the first is
    still being answered would be built without the first turn's context.

    If an earlier request hangs, we wait at most MEMORY_TURN_WAIT_S and then
    carry on rather than block the session forever."""
    if not session_id:
        yield
        return
    with _lock:
        gate = _gates.get(session_id)
        if gate is None:
            gate = _Gate()
            _gates[session_id] = gate
        gate.users += 1
    acquired = gate.lock.acquire(timeout=MEMORY_TURN_WAIT_S)
    try:
        yield
    finally:
        if acquired:
            gate.lock.release()
        with _lock:
            gate.users -= 1
            if gate.users <= 0 and _gates.get(session_id) is gate:
                _gates.pop(session_id, None)