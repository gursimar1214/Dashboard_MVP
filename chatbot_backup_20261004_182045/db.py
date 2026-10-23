"""
db.py -- stage 6, query execution.

Opens the connection as read-only at the SQLite driver level (a
`mode=ro` URI, plus PRAGMA query_only) rather than just trusting the
validated SQL -- a third, independent line of defense behind the
grammar constraint (stage 4) and the AST check (stage 5). A write
statement that somehow got this far would raise here, not silently run.

Also exposes `db_reference_date()`, the max transaction timestamp in the
dataset, used to resolve relative dates ("last month", "this year")
against *the data's* calendar rather than the wall clock. Cached with a
short TTL so a database populated after server start is picked up.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Any

from .config import DB_PATH, REFERENCE_DATE_TTL_S, SQL_QUERY_TIMEOUT_S

log = logging.getLogger("chatbot.db")

_FALLBACK_REFERENCE_DATE = "2024-09-26"

_ref_lock = threading.Lock()
_ref_cache: dict[str, Any] = {"value": None, "expires_at": 0.0}


class QueryExecutionError(Exception):
    pass


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise QueryExecutionError(f"no database found at '{DB_PATH}'")
    uri = f"file:{DB_PATH}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5.0)
    except sqlite3.OperationalError as exc:
        raise QueryExecutionError(f"could not open '{DB_PATH}': {exc}") from exc
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = TRUE;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA mmap_size = 268435456;")  # 256 MiB
    except sqlite3.Error:
        # pragmas are best-effort; a locked/legacy DB can still be queried
        pass
    return conn


def run_readonly_query(sql: str) -> list[dict[str, Any]]:
    conn = _connect()
    deadline = time.monotonic() + SQL_QUERY_TIMEOUT_S if SQL_QUERY_TIMEOUT_S > 0 else None
    if deadline is not None:
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10_000)
    try:
        cursor = conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        if "interrupted" in str(exc).lower() and deadline is not None:
            raise QueryExecutionError(
                f"query exceeded the {SQL_QUERY_TIMEOUT_S:g}-second time limit"
            ) from exc
        raise QueryExecutionError(f"query failed: {exc}") from exc
    finally:
        conn.close()


def db_reference_date() -> str:
    """Most recent transaction date in the dataset. Cached for
    REFERENCE_DATE_TTL_S seconds so it reflects a DB that was populated
    after the server started, without hitting the DB on every request."""
    now = time.time()
    with _ref_lock:
        if _ref_cache["value"] is not None and now < _ref_cache["expires_at"]:
            return _ref_cache["value"]

    value = _FALLBACK_REFERENCE_DATE
    try:
        rows = run_readonly_query(
            "SELECT MAX(transaction_timestamp) AS d FROM transactions"
        )
        if rows and rows[0].get("d"):
            value = str(rows[0]["d"])
    except QueryExecutionError as exc:
        log.warning("Could not read reference date (%s); using fallback %s.", exc, value)
    except Exception:
        log.exception("Unexpected error reading reference date; using fallback %s.", value)

    with _ref_lock:
        _ref_cache["value"] = value
        _ref_cache["expires_at"] = now + REFERENCE_DATE_TTL_S
    return value
