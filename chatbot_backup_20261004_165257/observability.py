"""
pipeline.py

Runs stages 2-7 for one user message. Stage 1 (FastAPI) lives in main.py
and just calls answer_question() below.

Two modes (env var CHATBOT_FAST_MODE, default "1"):

  FAST MODE (default): stage 2 (intent) and stage 4 (SQL generation) are
  done in ONE model call that returns {intent, sql}. Stage 7 never calls
  the model: a single number or a short list of rows is formatted in
  Python. Typical question = 1 LLM call.

  STANDARD MODE (CHATBOT_FAST_MODE=0): separate intent and SQL calls,
  and an LLM-written sentence for multi-row results.

Accuracy helpers (both modes):
  - few-shot examples in the prompt (different from the golden test set,
    so tests stay honest);
  - value_fix.canonicalize_values() puts text values into the exact
    spelling used in the database;
  - in fast mode, if the SQL fails validation or execution, the error is
    sent back to the model ONCE (CHATBOT_MAX_RETRIES) to correct it.

Follow-up memory (see memory.py, followup.py): the last few answered turns
of the session -- what was typed, what it meant, the SQL, the result -- are
shown to the model before the new question, and the model returns a
"standalone_question" (the follow-up rewritten to be self-contained) that is
stored as the turn's meaning. So "and last month?" after "what about india"
after "revenue in germany" is remembered as "revenue in india last month",
and chains of follow-ups stay accurate. A follow-up with no history to
follow is not guessed at: the person is asked to state it in full.
"""
from __future__ import annotations

import dataclasses
import hashlib
import uuid
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from . import fastpath, feedback, followup, llm, memory, slotmem, synonyms
from . import observability as obs
from .date_resolver import resolve_time_window
from .config import (
    CACHE_MAX_ENTRIES,
    CACHE_TTL_S,
    FAST_MODE,
    INTENT_MAX_TOKENS,
    MAX_LISTED_ROWS,
    MAX_RETRIES,
    NL_MAX_TOKENS,
    SQL_MAX_TOKENS,
)
from .db import QueryExecutionError, db_reference_date, run_readonly_query
from .intent_schema import ExtractedQuery, Intent
from .schema_glossary import (
    COMMON_MISTAKES,
    METRICS_GLOSSARY,
    SCHEMA_DESCRIPTION,
    VALID_VALUES_TEXT,
)
from .sql_schema import GeneratedSQL
from .sql_validator import SQLValidationError, validate_and_cap
from .value_fix import canonicalize_values

log = logging.getLogger("chatbot.pipeline")

ProgressFn = Callable[[str], None]


# --------------------------------------------------------------------------
# Prompts (built once at import; Ollama caches the prefix after warmup)
# --------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    ("how many gold customers are in france",
     "SELECT COUNT(*) AS customers FROM customers WHERE country_code = 'FR' AND customer_tier = 'Gold'"),
    ("number of orders in france and germany",
     "SELECT c.country_code, COUNT(*) AS orders FROM transactions t JOIN customers c ON c.customer_id = t.customer_id WHERE c.country_code IN ('FR', 'DE') GROUP BY c.country_code"),
    ("how many sessions did french customers have",
     "SELECT COUNT(*) AS sessions FROM sessions s JOIN customers c ON c.customer_id = s.customer_id WHERE c.country_code = 'FR'"),
    ("top customer by number of purchases in germany",
     "SELECT c.customer_id, COUNT(*) AS purchases FROM customers c JOIN transactions t ON t.customer_id = c.customer_id WHERE c.country_code = 'DE' GROUP BY c.customer_id ORDER BY purchases DESC LIMIT 1"),
    ("average order value for silver customers",
     "SELECT SUM(t.order_value) / COUNT(*) AS aov FROM transactions t JOIN customers c ON c.customer_id = t.customer_id WHERE c.customer_tier = 'Silver'"),
    ("orders by shipping speed",
     "SELECT shipping_speed, COUNT(*) AS orders FROM transactions GROUP BY shipping_speed ORDER BY orders DESC"),
    ("monthly orders in 2022",
     "SELECT substr(transaction_timestamp, 1, 7) AS month, COUNT(*) AS orders FROM transactions WHERE transaction_timestamp >= '2022-01-01' AND transaction_timestamp < '2023-01-01' GROUP BY month ORDER BY month"),
    ("bounce rate by device type",
     "SELECT device_type, AVG(CASE WHEN bounce_flag THEN 1.0 ELSE 0.0 END) AS bounce_rate FROM sessions GROUP BY device_type"),
    ("how many customers have no sessions",
     "SELECT COUNT(*) AS customers FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM sessions)"),
    # Derived table (top-N per group): the validator accepts these since the audit fix.
    ("top 2 customers by revenue in each country",
     "SELECT country_code, customer_id, revenue FROM (SELECT c.country_code, c.customer_id, SUM(t.order_value) AS revenue, ROW_NUMBER() OVER (PARTITION BY c.country_code ORDER BY SUM(t.order_value) DESC) AS rn FROM customers c JOIN transactions t ON t.customer_id = c.customer_id GROUP BY c.country_code, c.customer_id) AS ranked WHERE rn <= 2 ORDER BY country_code, rn"),
    # Campaign join: sessions.campaign_id always matches a campaign, so a plain JOIN is right.
    ("how many sessions came from display campaigns",
     "SELECT COUNT(*) AS sessions FROM sessions s JOIN marketing_campaigns m ON m.campaign_id = s.campaign_id WHERE m.campaign_type = 'Display'"),
]
FEW_SHOT_TEXT = "\n\n".join(f"Q: {q}\nSQL: {sql}" for q, sql in FEW_SHOT_EXAMPLES)

CONTEXT_INSTRUCTION = (
    "The conversation so far is shown below, oldest first. Decide whether the "
    "NEW question is a follow-up that depends on it, e.g. \"and last month?\", "
    "\"what about india\", \"same for gold customers\", \"break it down by "
    "device\", \"only mobile\", \"top 10 instead\".\n"
    "If it IS a follow-up, build on the MOST RECENT turn (unless the question "
    "clearly points to an earlier one) using these rules:\n"
    "1. KEEP the same metric (same aggregate and SELECTed columns), the same "
    "joins and every filter; change ONLY what the new question says. A "
    "follow-up that only names a new period or a new filter value must NOT "
    "switch the metric: revenue stays SUM(order_value) and never becomes a "
    "row count, unless the new question itself asks for a count.\n"
    "2. A new value for a filter that already exists REPLACES it (india "
    "replaces germany, 2023 replaces 2022, top 10 replaces top 5). A filter "
    "on something not filtered yet is ADDED. \"overall\", \"all countries\", "
    "\"everyone\" REMOVES that filter. \"by X\" / \"break it down by X\" keeps "
    "the metric and filters and adds GROUP BY X.\n"
    "3. Relative dates (\"last month\", \"this year\") are measured from the "
    "reference date, NOT from the dates in the earlier query. \"the month "
    "before\", \"previous period\", \"the year before\" shift the earlier "
    "query's date range.\n"
    "If the new question is NOT a follow-up (it names its own metric and "
    "scope), ignore the conversation completely: none of the earlier filters "
    "apply."
)

# Worked follow-ups. Shown only when the session has history, so first
# questions keep the shorter prompt. Deliberately different from the golden
# test questions. Every SQL here is executed against the schema by the tests.
FOLLOWUP_BASE_QUESTION = "total revenue in germany in 2022"
FOLLOWUP_BASE_SQL = (
    "SELECT SUM(t.order_value) AS revenue FROM transactions t "
    "JOIN customers c ON c.customer_id = t.customer_id "
    "WHERE c.country_code = 'DE' AND t.transaction_timestamp >= '2022-01-01' "
    "AND t.transaction_timestamp < '2023-01-01'"
)
FOLLOWUP_EXAMPLES_LIST = [
    ("what about india",
     "total revenue in india in 2022",
     "SELECT SUM(t.order_value) AS revenue FROM transactions t "
     "JOIN customers c ON c.customer_id = t.customer_id "
     "WHERE c.country_code = 'IN' AND t.transaction_timestamp >= '2022-01-01' "
     "AND t.transaction_timestamp < '2023-01-01'"),
    ("and only gold customers?",
     "total revenue in germany in 2022 from gold customers",
     "SELECT SUM(t.order_value) AS revenue FROM transactions t "
     "JOIN customers c ON c.customer_id = t.customer_id "
     "WHERE c.country_code = 'DE' AND c.customer_tier = 'Gold' "
     "AND t.transaction_timestamp >= '2022-01-01' AND t.transaction_timestamp < '2023-01-01'"),
    ("and 2023?",
     "total revenue in germany in 2023",
     "SELECT SUM(t.order_value) AS revenue FROM transactions t "
     "JOIN customers c ON c.customer_id = t.customer_id "
     "WHERE c.country_code = 'DE' AND t.transaction_timestamp >= '2023-01-01' "
     "AND t.transaction_timestamp < '2024-01-01'"),
    ("break it down by payment method",
     "total revenue in germany in 2022 by payment method",
     "SELECT t.payment_method, SUM(t.order_value) AS revenue FROM transactions t "
     "JOIN customers c ON c.customer_id = t.customer_id "
     "WHERE c.country_code = 'DE' AND t.transaction_timestamp >= '2022-01-01' "
     "AND t.transaction_timestamp < '2023-01-01' GROUP BY t.payment_method ORDER BY revenue DESC"),
    ("across all countries",
     "total revenue across all countries in 2022",
     "SELECT SUM(t.order_value) AS revenue FROM transactions t "
     "WHERE t.transaction_timestamp >= '2022-01-01' AND t.transaction_timestamp < '2023-01-01'"),
    ("how many sessions came from ads",
     "how many sessions came from ads",
     "SELECT COUNT(*) AS sessions FROM sessions WHERE traffic_source = 'ads'"),
]
FOLLOWUP_EXAMPLES = (
    "Worked examples. Suppose the most recent turn was "
    f"\"{FOLLOWUP_BASE_QUESTION}\" and it ran:\n{FOLLOWUP_BASE_SQL}\n"
    "Then, for each possible NEW question:\n"
    + "\n".join(
        f"- New question: \"{q}\"\n  standalone_question: {sq}\n  sql: {sql}"
        for q, sq, sql in FOLLOWUP_EXAMPLES_LIST
    )
    + "\n(The last example is NOT a follow-up: it names its own metric, so the earlier filters are ignored.)"
)

INTENT_SYSTEM_PROMPT = f"""
You extract structured intent from a question about an ecommerce
dashboard. The dashboard only has these tables:

{SCHEMA_DESCRIPTION}

{VALID_VALUES_TEXT}

Classify the question into exactly one intent (overview, trend,
breakdown, lookup, ranking, comparison, followup, out_of_scope) and pull
out any filters the user actually mentioned (country, tier, channel,
date range, one customer_id, a top-N count). Use "followup" only if the
question depends on an earlier turn that is NOT shown to you. Use
"out_of_scope" for questions that can't be answered from these tables at
all. Never invent a filter value the user didn't mention.

Also write "standalone_question": the question rewritten so it can be
understood with no conversation history, spelling out any metric, filters
and period carried over from the conversation shown. If the question is
already self-contained, repeat it. Fill in the filters from the
conversation too, following the follow-up rules given with it.
""".strip()

SQL_SYSTEM_PROMPT = f"""
You write exactly one read-only SQLite SELECT statement based on a
structured request, and nothing else.

Schema:
{SCHEMA_DESCRIPTION}

{VALID_VALUES_TEXT}

Metric definitions -- use these exact formulas, never invent your own:
{METRICS_GLOSSARY}

Rules:
{COMMON_MISTAKES}

Examples (question -> SQL):
{FEW_SHOT_TEXT}
""".strip()

FAST_SYSTEM_PROMPT = f"""
You answer questions about an ecommerce dashboard database by writing
exactly one read-only SQLite SELECT statement.

Schema:
{SCHEMA_DESCRIPTION}

{VALID_VALUES_TEXT}

Metric definitions -- use these exact formulas, never invent your own:
{METRICS_GLOSSARY}

Rules:
{COMMON_MISTAKES}

Examples (question -> SQL):
{FEW_SHOT_TEXT}

Follow-ups: if a "Conversation so far" block is shown and the NEW question
depends on it ("and last month?", "what about india", "only gold", "by
device"), keep the previous metric, joins and filters and change ONLY what
the new question says. A new value for an existing filter replaces it; a
new kind of filter is added; "overall"/"all countries" removes the filter;
"by X" adds GROUP BY X. If the NEW question names its own metric, ignore
the conversation. Relative dates are measured from the reference date.

A note in parentheses after the question says exactly which column or table
is meant; follow it.

Reply as JSON.
- With a "Conversation so far" block: {{"standalone_question": the NEW
  question rewritten to be self-contained, "sql": ...}}
- Without one: {{"sql": ...}}
"sql" is null when the question cannot be answered from these tables.
""".strip()

NL_SYSTEM_PROMPT = """
You explain query results from an ecommerce dashboard in plain
language. State only numbers present in the rows you're given -- never
estimate, round dramatically, or add a number that isn't there. Lead
with the direct answer in one sentence, then at most one sentence of
context. Use natural phrasing: "Revenue in Germany was $1.2M."
""".strip()

OUT_OF_SCOPE_REPLY = (
    "That's outside what this dashboard tracks -- I can only answer questions "
    "about customers, sessions, transactions, geo data, and marketing campaigns."
)
NO_DATA_REPLY = "No data matched those filters -- try loosening one of them."
NEEDS_CONTEXT_REPLY = (
    "That reads like a follow-up, but I don't have an earlier question in this "
    "chat to build on. Could you ask it in full? For example: \"revenue in "
    "India last month\"."
)
UNCLEAR_FOLLOWUP_REPLY = (
    "I couldn't tell what that refers to. Could you ask it in full? For "
    "example: \"revenue in India last month\"."
)

_MONEY_KEY_HINTS = ("revenue", "aov", "order_value", "credit_limit", "budget", "income", "spend", "value")
_RATE_KEY_HINTS = ("rate", "share", "ratio", "penetration")
_RAW_KEY_HINTS = ("_id", "year", "month", "date", "day")


class FastPlan(BaseModel):
    # Used when there is conversation history: the model states the
    # self-contained question BEFORE writing the SQL for it.
    standalone_question: str = Field(
        ...,
        description="The question rewritten to be understandable with no conversation history.",
    )
    sql: Optional[str] = Field(
        default=None,
        description="One read-only SQLite SELECT statement; null if not answerable.",
    )


class FastPlanNoCtx(BaseModel):
    # First question of a chat: SQL only. Fewer output tokens = faster
    # (the model writes ~11 tokens/s on an iGPU, so every token costs).
    sql: Optional[str] = Field(
        default=None,
        description="One read-only SQLite SELECT statement; null if not answerable.",
    )


def _guess_intent(sql: str) -> Intent:
    """Label for the API/memory, derived from the SQL in Python (costs no
    model tokens)."""
    s = sql.lower()
    if "substr(" in s and "group by" in s:
        return Intent.TREND
    if "group by" in s:
        return Intent.BREAKDOWN
    if "order by" in s and "limit" in s:
        return Intent.RANKING
    # a lookup names ONE customer (customer_id = 'C123' / = 42); a JOIN condition
    # such as "c.customer_id = t.customer_id" must not count.
    if re.search(r"customer_id\s*=\s*(?:'[^']*'|\d+)", s) and "group by" not in s \
            and "count(" not in s and "sum(" not in s:
        return Intent.LOOKUP
    return Intent.OVERVIEW


@dataclass
class ChatAnswer:
    text: str
    sql: Optional[str]
    rows: Optional[list[dict[str, Any]]]
    intent: str
    # The question as the model understood it once follow-up references were
    # resolved, and the SQL as written (before the row cap was injected).
    # Both feed the session memory; neither is required by the API.
    resolved_question: Optional[str] = None
    memory_sql: Optional[str] = None
    interpreted_as: Optional[str] = None   # shown to the user only when it differs from what they typed
    answer_id: Optional[str] = None        # id the dashboard sends back with thumbs up/down

    def to_dict(self) -> dict:
        return {
            "answer": self.text, "sql": self.sql, "rows": self.rows,
            "intent": self.intent, "interpreted_as": self.interpreted_as,
            "answer_id": self.answer_id,
        }


# --------------------------------------------------------------------------
# Formatting (pure Python, no model)
# --------------------------------------------------------------------------

def _pretty_value(key: str, value: Any) -> str:
    if value is None:
        return "n/a"
    key_l = key.lower()
    if any(hint in key_l for hint in _RAW_KEY_HINTS):
        return str(value)
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if any(hint in key_l for hint in _RATE_KEY_HINTS):
        return f"{num:.1%}" if abs(num) <= 1 else f"{num:.2f}"
    if any(hint in key_l for hint in _MONEY_KEY_HINTS):
        return f"${num:,.2f}"
    if num.is_integer():
        return f"{int(num):,}"
    return f"{num:,.2f}"


def _is_numericish(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _label_column(row: dict[str, Any]) -> str:
    """Pick which column is the row's label in a multi-row result.
    Prefer the first non-numeric, non-id-looking column; fall back to the
    first column."""
    for k, v in row.items():
        kl = k.lower()
        if "_id" in kl or kl in {"year", "month", "date", "day"}:
            continue
        if not _is_numericish(v):
            return k
    return next(iter(row))


def _format_overview_reply(row: dict[str, Any]) -> str:
    # Single-KPI row (one column, one value) -> direct sentence.
    if len(row) == 1:
        (key, value), = row.items()
        label = {
            "aov": "Average order value",
            "avg_email_open_rate": "Average email open rate",
        }.get(key.lower(), key.replace("_", " ").capitalize())
        return f"{label} is {_pretty_value(key, value)}."
    parts = [f"{key.replace('_', ' ')}: {_pretty_value(key, value)}" for key, value in row.items()]
    return "Here's what the data shows -- " + "; ".join(parts) + "."


def _format_rows_reply(rows: list[dict[str, Any]]) -> str:
    if len(rows) == 1:
        return _format_overview_reply(rows[0])
    label_col = _label_column(rows[0])
    lines = []
    for row in rows[:MAX_LISTED_ROWS]:
        label = str(row.get(label_col, ""))
        rest = ", ".join(
            f"{k.replace('_', ' ')}: {_pretty_value(k, v)}"
            for k, v in row.items() if k != label_col
        )
        lines.append(f"- {label}" + (f" -- {rest}" if rest else ""))
    suffix = f", first {MAX_LISTED_ROWS} listed" if len(rows) > MAX_LISTED_ROWS else ""
    return f"Here's what the data shows ({len(rows)} rows{suffix}):\n" + "\n".join(lines)


def _mem_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) >= 1 or v == 0 else f"{v:.4f}"
    return str(v)


def _summarise_for_memory(rows: list[dict[str, Any]]) -> str:
    """One-line, model-readable description of a result, for the memory
    store. Says what came back (and how much), not just the first cell."""
    if not rows:
        return "no rows (the filters matched nothing)"
    if all(v is None for row in rows for v in row.values()):
        return "no data (every value was empty)"

    def fmt(row: dict[str, Any]) -> str:
        return ", ".join(f"{k}={_mem_value(v)}" for k, v in row.items())

    if len(rows) == 1:
        return fmt(rows[0])
    shown = "; ".join(fmt(r) for r in rows[:3])
    more = "" if len(rows) <= 3 else f"; ... ({len(rows) - 3} more)"
    return f"{len(rows)} rows, first: {shown}{more}"


# --------------------------------------------------------------------------
# Cache (keyed by question + session turn fingerprint)
# --------------------------------------------------------------------------

_cache: "OrderedDict[str, tuple[float, ChatAnswer]]" = OrderedDict()
_cache_lock = threading.Lock()


def _normalise(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _cache_key(question: str, session_fingerprint: str) -> str:
    h = hashlib.sha1(session_fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"{_normalise(question)}||{h}"


def _cache_get(key: str) -> Optional[ChatAnswer]:
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if now - stored_at > CACHE_TTL_S:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return value


def _cache_put(key: str, value: ChatAnswer) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), value)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)


# --------------------------------------------------------------------------
# Warmup
# --------------------------------------------------------------------------

def warmup() -> None:
    """Load the model and pre-process the long system prompt once at server
    start (one generated token; the prompt prefix stays cached in Ollama)."""
    llm.warmup(FAST_SYSTEM_PROMPT if FAST_MODE else SQL_SYSTEM_PROMPT)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def answer_question(
    question: str,
    session_id: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> ChatAnswer:
    """One chat request = one Langfuse trace (when tracing is on) and one
    answer_id the dashboard can attach feedback to (always)."""
    info: dict[str, Any] = {"path": "llm"}
    with obs.request_trace(
        "chat-request", session_id=session_id, input=question,
        tags=[f"model:{llm.LLM_MODEL}", "mode:fast" if FAST_MODE else "mode:standard"],
    ) as root:
        result = _answer_question(question, session_id, progress, info)
        trace_id = root.trace_id
        answer_id = trace_id or uuid.uuid4().hex
        result = dataclasses.replace(result, answer_id=answer_id)
        feedback.remember_answer(
            answer_id, trace_id=trace_id, session_id=session_id, question=question,
            understood=result.interpreted_as or result.resolved_question, sql=result.sql,
            answer=result.text, intent=result.intent, path=info.get("path"),
            resolution=info.get("resolution"), model=llm.LLM_MODEL,
        )
        root.update(
            output={
                "answer": result.text, "intent": result.intent, "sql": result.sql,
                "understood_as": result.interpreted_as, "result": obs.trim_rows(result.rows),
            },
            metadata={**info, "model": llm.LLM_MODEL, "fast_mode": FAST_MODE},
        )
        if trace_id:
            # Filterable in Langfuse on every SDK version (tags can't be set late in v4).
            obs.score(trace_id, "answered_from_data", 1.0 if result.rows else 0.0)
            obs.score(trace_id, "path", str(info.get("path")))
    return result


def _answer_question(
    question: str,
    session_id: Optional[str],
    progress: Optional[ProgressFn],
    info: dict,
) -> ChatAnswer:
    def emit(stage: str) -> None:
        if progress is not None:
            try:
                progress(stage)
            except Exception:
                log.exception("progress callback raised")

    # One turn at a time per session: a second message sent while the first is
    # still being answered must see the first one's result as context.
    with memory.turn_gate(session_id):
        context, fingerprint = memory.prompt_context(session_id)   # read together

        # Synonym layer: typo-fixed text + what Python recognised (no model).
        analysis = synonyms.analyze(question)

        # "and last month?" with nothing to follow: ask, don't guess (and
        # don't let a guess become the memory later follow-ups build on).
        if followup.needs_earlier_turn(analysis.text, has_context=bool(context)):
            log.info("dangling follow-up with no history: %r", question)
            info["path"] = "followup-guard"
            return ChatAnswer(
                text=NEEDS_CONTEXT_REPLY, sql=None, rows=None, intent=Intent.FOLLOWUP.value
            )

        # Slot memory: is this a NEW question, a follow-up Python fully
        # understands (merge slots, no history sent), or one it leaves to the
        # model (legacy: history text, as before)?
        prev_slots = slotmem.Slots.from_dict(memory.last_slots(session_id))
        res = slotmem.resolve(question, prev_slots, has_history=bool(context), analysis=analysis)
        info["resolution"] = res.mode
        info["resolution_reason"] = res.reason
        if analysis.corrections:
            info["typo_fixes"] = analysis.corrections
        if res.hints:
            info["hints"] = res.hints
        if res.mode == "slots":
            info["slots"] = res.slots.to_dict() if res.slots else None
        log.info("resolution=%s (%s) question=%r -> %r", res.mode, res.reason, question, res.model_question)

        # Exact templates take priority over the model, even when this is a
        # complete new question inside an existing chat. Some follow-ups
        # also become fully answerable after their slots have been merged.
        fixed_sql = None
        if FAST_MODE:
            slots = res.slots
            ref_date = db_reference_date() if slots and slots.time else None
            if slots:
                fixed_sql = fastpath.match_slots(
                    slots.metric, slots.dimension, slots.filters, slots.time,
                    slots.rank, slots.top_n, ref_date,
                )
            if fixed_sql is None:
                fixed_sql = fastpath.match(analysis.text)

        use_history = res.use_history and fixed_sql is None
        use_context = context if use_history else ""
        use_fp = fingerprint if use_history else ""
        key = _cache_key(res.model_question + "|" + "|".join(res.hints), use_fp)
        emit("cache")
        cached = _cache_get(key)
        if cached is not None:
            log.info("cache hit: %s", key)
            info["path"] = "cache"
            result = cached
        else:
            result = _answer_uncached(
                res.model_question, use_context, progress=emit, info=info,
                hints=res.hints, fixed_sql=fixed_sql,
            )
            # Only cache genuinely successful, non-empty answers.
            if result.rows and result.intent != Intent.OUT_OF_SCOPE.value:
                _cache_put(key, result)

        # In fresh/slots mode the model never saw history, so what it "resolved"
        # is Python's reading. Applied after the cache so a cached answer shows
        # the right "Understood as" for THIS typed question.
        if not use_history:
            understood = res.resolved_text or res.model_question
            shown = understood if _normalise(understood) != _normalise(question) else None
            if res.mode == "fresh" and not analysis.corrections:
                shown = None
            result = dataclasses.replace(result, resolved_question=understood, interpreted_as=shown)

        # Remember every turn whose SQL ran -- including ones that matched no
        # rows -- so the next follow-up has something to attach to.
        if result.sql is not None and result.rows is not None:
            turn_slots = res.slots if not use_history else slotmem.slots_from_text(result.resolved_question)
            memory.remember(
                session_id,
                question,
                result.memory_sql or result.sql,
                result.intent,
                summary=_summarise_for_memory(result.rows),
                resolved_question=result.resolved_question or "",
                slots=turn_slots.to_dict() if turn_slots else None,
            )
    return result


# --------------------------------------------------------------------------
# The pipeline itself
# --------------------------------------------------------------------------

def _retry_prompt(base_prompt: str, previous_sql: str, error: str) -> str:
    """Send the model its own failure. `base_prompt` is the ORIGINAL prompt
    (conversation context + reference date + question) so the retry keeps
    everything the first attempt had."""
    return (
        f"{base_prompt}\n\n"
        f"Your previous SQL was:\n{previous_sql}\n\n"
        f"It failed with: {error}\n\n"
        f"Reply with the corrected JSON."
    )


def _plan_fast(prompt: str, has_context: bool):
    return llm.generate_structured(
        system=FAST_SYSTEM_PROMPT, prompt=prompt,
        schema=FastPlan if has_context else FastPlanNoCtx,
        max_tokens=SQL_MAX_TOKENS, name="sql-plan",
    )


def _answer_uncached(
    question: str,
    context: str = "",
    progress: Optional[ProgressFn] = None,
    info: Optional[dict] = None,
    hints: Optional[list[str]] = None,
    fixed_sql: Optional[str] = None,
) -> ChatAnswer:
    if info is None:
        info = {}

    def emit(stage: str) -> None:
        if progress is not None:
            try:
                progress(stage)
            except Exception:
                log.exception("progress callback raised")

    total_start = time.perf_counter()
    timings: dict[str, int] = {}
    intent: Intent = Intent.OVERVIEW
    generated_sql: Optional[str] = None

    reference_date = db_reference_date()
    context_block = (
        f"Conversation so far:\n{context}\n\n"
        if context else ""
    )
    q_label = "NEW question" if context else "Question"
    # Synonym-layer hints: which column an ambiguous word means. Shown after
    # the question so the model cannot miss them.
    hint_block = ("\n(" + "; ".join(hints) + ")") if hints else ""
    time_phrase = synonyms.analyze(question).time
    date_window = resolve_time_window(time_phrase, reference_date) if time_phrase else None
    if date_window:
        info["relative_date_window"] = {
            "phrase": time_phrase, "reference_date": reference_date,
            "start_inclusive": date_window.start.isoformat(),
            "end_exclusive": date_window.end.isoformat(),
        }
        sql_rules_inline = (
            f"Latest date in the dataset: {reference_date}. "
            f"Relative period rule: {date_window.instruction()} Use these exact bounds "
            "for transaction_timestamp; do not infer a different year."
        )
    else:
        sql_rules_inline = f"Reference date (treat as today for relative dates): {reference_date}"
    resolved = question            # what the question means once made self-contained

    def _interpreted(text: str) -> Optional[str]:
        """Show the person how a follow-up was read -- only when there was
        history and the reading differs from what they typed."""
        if context and text and _normalise(text) != _normalise(question):
            return text
        return None

    # ---- STANDARD MODE: separate intent call, then SQL call --------------
    if not FAST_MODE:
        emit("intent")
        t = time.perf_counter()
        try:
            extracted = llm.generate_structured(
                system=INTENT_SYSTEM_PROMPT,
                prompt=f"{context_block}{sql_rules_inline}\n\n{q_label}: {question}{hint_block}",
                schema=ExtractedQuery,
                max_tokens=INTENT_MAX_TOKENS, name="intent",
            )
        except llm.LLMError as exc:
            log.warning("stage2 intent extraction failed: %s", exc)
            return ChatAnswer(text=exc.user_message, sql=None, rows=None, intent="error")
        timings["llm_intent"] = _ms(t)
        intent = extracted.intent
        resolved = _clean_resolved(extracted.standalone_question, question)
        log.info("stage2 intent=%s entities=%s", intent, extracted.model_dump(exclude_none=True))
        if intent == Intent.OUT_OF_SCOPE:
            return ChatAnswer(text=OUT_OF_SCOPE_REPLY, sql=None, rows=None, intent=intent.value)
        if intent == Intent.FOLLOWUP and followup.needs_earlier_turn(
            question, has_context=bool(context), model_says_followup=True
        ):
            return ChatAnswer(text=NEEDS_CONTEXT_REPLY, sql=None, rows=None, intent=intent.value)
        if intent == Intent.FOLLOWUP and context:
            log.info("stage2: model flagged follow-up %r with history present", question)

        emit("sql")
        sql_prompt = (
            f"{context_block}{sql_rules_inline}\n\n"
            f"Structured request:\n{extracted.model_dump_json(exclude_none=True)}\n\n"
            f"Original question (for context only -- the structured request above "
            f"is authoritative for filters): {question}"
        )
        t = time.perf_counter()
        try:
            generated = llm.generate_structured(
                system=SQL_SYSTEM_PROMPT, prompt=sql_prompt,
                schema=GeneratedSQL, max_tokens=SQL_MAX_TOKENS, name="sql",
            )
        except llm.LLMError as exc:
            log.warning("stage4 SQL generation failed: %s", exc)
            return ChatAnswer(text=exc.user_message, sql=None, rows=None, intent=intent.value)
        timings["llm_sql"] = _ms(t)
        generated_sql = generated.sql

    attempts = 1 + (MAX_RETRIES if FAST_MODE else 0)
    base_prompt = f"{context_block}{sql_rules_inline}\n\n{q_label}: {question}{hint_block}"
    prompt = base_prompt
    safe_sql = ""
    memory_sql = ""
    rows: list[dict[str, Any]] = []

    # Simple, unambiguous KPI question with no chat history: fixed SQL, no model.
    fast_sql = fixed_sql
    if fast_sql is None and FAST_MODE and not context:
        fast_sql = fastpath.match(question)

    for attempt in range(1, attempts + 1):
        if fast_sql is not None:
            generated_sql, intent = fast_sql, _guess_intent(fast_sql)
            fast_sql = None                      # a retry (if any) goes to the model
            timings["fastpath"] = 0
            info["path"] = "fastpath"
        elif FAST_MODE:
            emit("plan")
            t = time.perf_counter()
            try:
                plan = _plan_fast(prompt, bool(context))
            except llm.LLMError as exc:
                log.warning("fast plan (SQL) failed: %s", exc)
                return ChatAnswer(text=exc.user_message, sql=None, rows=None, intent="error")
            timings[f"llm_plan_{attempt}"] = _ms(t)
            generated_sql = (plan.sql or "").strip() or None
            if context:
                resolved = _clean_resolved(plan.standalone_question, question)
            if not generated_sql:
                return ChatAnswer(
                    text=OUT_OF_SCOPE_REPLY, sql=None, rows=None,
                    intent=Intent.OUT_OF_SCOPE.value,
                )
            intent = _guess_intent(generated_sql)

        candidate = canonicalize_values(generated_sql)
        last_try = attempt == attempts
        info["attempts"] = attempt

        emit("validate")
        t = time.perf_counter()
        try:
            with obs.observation("validate", input=candidate, metadata={"attempt": attempt}) as vs:
                safe_sql = validate_and_cap(candidate)
                vs.update(output=safe_sql)
        except SQLValidationError as exc:
            info["validation_error"] = str(exc)[:300]
            log.warning("stage5 rejected SQL (attempt %d, %s): %s", attempt, exc, candidate)
            if not last_try:
                prompt = _retry_prompt(base_prompt, candidate, str(exc))
                continue
            return ChatAnswer(
                text=f"I generated a query that didn't pass validation ({exc}) -- try asking a bit differently.",
                sql=candidate, rows=None, intent=intent.value,
                resolved_question=resolved, interpreted_as=_interpreted(resolved),
            )
        timings[f"validate_{attempt}"] = _ms(t)

        emit("execute")
        t = time.perf_counter()
        try:
            with obs.observation("execute", input=safe_sql, metadata={"attempt": attempt}) as es:
                rows = run_readonly_query(safe_sql)
                es.update(output={"row_count": len(rows)})
        except QueryExecutionError as exc:
            info["execution_error"] = str(exc)[:300]
            log.warning("stage6 execution failed (attempt %d, %s): %s", attempt, exc, safe_sql)
            if not last_try:
                prompt = _retry_prompt(base_prompt, safe_sql, str(exc))
                continue
            return ChatAnswer(
                text=f"That query didn't run cleanly against the database ({exc}).",
                sql=safe_sql, rows=None, intent=intent.value,
                resolved_question=resolved, interpreted_as=_interpreted(resolved),
            )
        timings[f"execute_{attempt}"] = _ms(t)
        memory_sql = candidate      # as written, before the row cap was added
        break

    # ---- Stage 7: build the reply ---------------------------------------
    emit("reply")
    if not rows or all(v is None for row in rows for v in row.values()):
        text = NO_DATA_REPLY
    elif FAST_MODE:
        text = _format_rows_reply(rows)
    elif intent == Intent.OVERVIEW and len(rows) == 1:
        text = _format_overview_reply(rows[0])
    else:
        t = time.perf_counter()
        nl_prompt = (
            f"Question: {question}\n\n"
            f"Query results (JSON, {len(rows)} row(s)):\n{json.dumps(rows, default=str)[:6000]}"
        )
        try:
            text = llm.generate_text(system=NL_SYSTEM_PROMPT, prompt=nl_prompt, max_tokens=NL_MAX_TOKENS)
        except llm.LLMError as exc:
            # The data is fine; only the wording step failed -- format in Python.
            log.warning("stage7 reply generation failed (%s); formatting in Python.", exc)
            text = _format_rows_reply(rows)
        timings["llm_reply"] = _ms(t)

    info["timings_ms"] = {**timings, "total": _ms(total_start)}
    log.info(
        "timings ms: %s | total=%d | mode=%s",
        timings, _ms(total_start), "fast" if FAST_MODE else "standard",
    )
    return ChatAnswer(
        text=text, sql=safe_sql, rows=rows, intent=intent.value,
        resolved_question=resolved, memory_sql=memory_sql,
        interpreted_as=_interpreted(resolved),
    )


def _clean_resolved(text: Optional[str], fallback: str) -> str:
    """The model's standalone_question, or the typed question if it left it
    empty. One line, capped: it is stored and shown back to the model."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    return cleaned[:300] if cleaned else fallback


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
