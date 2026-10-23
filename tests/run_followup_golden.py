"""
Live follow-up accuracy test -- needs Ollama running with the model pulled
and the real database in place.

    python tests/run_followup_golden.py            # run everything
    python tests/run_followup_golden.py -v         # show SQL for every step
    python tests/run_followup_golden.py --min 0.9  # exit 1 if accuracy < 90%

Each conversation is a series of messages sent in ONE session, like a person
chatting. After every message the rows the chatbot returned are compared
with the rows of a hand-written REFERENCE query that expresses what the
message should mean in context. That checks the answer, not the wording of
the SQL, so any correct way of writing it passes.

Comparison is deliberately tolerant about presentation and strict about
substance: column names/aliases are ignored, label columns (text) are
ignored, row order is ignored, numbers are compared to 2 decimals. What must
match is every number, in the right number of rows.

Not covered on purpose: relative dates ("last month"). Whether that means
the last calendar month or the last 30 days is a product decision, and the
reference answer would just restate the decision. Add your own once decided.

Start from the project root (the folder containing chatbot/).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import db, memory, pipeline  # noqa: E402

ASK = "__expect_clarification__"

_DE22 = ("c.country_code = 'DE' AND t.transaction_timestamp >= '2022-01-01' "
         "AND t.transaction_timestamp < '2023-01-01'")
_J = "FROM transactions t JOIN customers c ON c.customer_id = t.customer_id"


def _rev(where: str, group: str = "") -> str:
    sel = f"{group}, " if group else ""
    tail = f" GROUP BY {group}" if group else ""
    return f"SELECT {sel}SUM(t.order_value) AS revenue {_J} WHERE {where}{tail}"


def _cnt(where: str, group: str = "") -> str:
    sel = f"{group}, " if group else ""
    tail = f" GROUP BY {group}" if group else ""
    return f"SELECT {sel}COUNT(*) AS n {_J} WHERE {where}{tail}"


def _y(year: int) -> str:
    return f"t.transaction_timestamp >= '{year}-01-01' AND t.transaction_timestamp < '{year + 1}-01-01'"


CONVERSATIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("replace / add / replace / group / remove", [
        ("total revenue in germany in 2022", _rev(_DE22)),
        ("what about india", _rev(f"c.country_code = 'IN' AND {_y(2022)}")),
        ("and only gold customers?", _rev(f"c.country_code = 'IN' AND c.customer_tier = 'Gold' AND {_y(2022)}")),
        ("and 2023?", _rev(f"c.country_code = 'IN' AND c.customer_tier = 'Gold' AND {_y(2023)}")),
        ("break it down by payment method",
         _rev(f"c.country_code = 'IN' AND c.customer_tier = 'Gold' AND {_y(2023)}", "t.payment_method")),
        ("across all countries",
         _rev(f"c.customer_tier = 'Gold' AND {_y(2023)}", "t.payment_method")),
    ]),
    ("a count stays a count; an explicit metric change is honoured", [
        ("how many orders in france", _cnt("c.country_code = 'FR'")),
        ("and germany?", _cnt("c.country_code = 'DE'")),
        ("what about platinum customers", _cnt("c.country_code = 'DE' AND c.customer_tier = 'Platinum'")),
        ("what about their revenue?", _rev("c.country_code = 'DE' AND c.customer_tier = 'Platinum'")),
    ]),
    ("sessions table, filters on the same table", [
        ("how many sessions came from ads",
         "SELECT COUNT(*) FROM sessions WHERE traffic_source = 'ads'"),
        ("and from email?",
         "SELECT COUNT(*) FROM sessions WHERE traffic_source = 'email'"),
        ("only on mobile",
         "SELECT COUNT(*) FROM sessions WHERE traffic_source = 'email' AND device_type = 'mobile'"),
        ("what about tablet",
         "SELECT COUNT(*) FROM sessions WHERE traffic_source = 'email' AND device_type = 'tablet'"),
    ]),
    ("average order value, filters accumulate", [
        ("average order value by payment method",
         "SELECT payment_method, AVG(order_value) FROM transactions GROUP BY payment_method"),
        ("only for gold customers",
         _rev("c.customer_tier = 'Gold'", "t.payment_method").replace("SUM(t.order_value)", "AVG(t.order_value)")),
        ("and only in germany",
         _rev("c.customer_tier = 'Gold' AND c.country_code = 'DE'", "t.payment_method")
         .replace("SUM(t.order_value)", "AVG(t.order_value)")),
    ]),
    ("period follow-up on a grouped count", [
        ("orders by shipping speed in 2022",
         "SELECT shipping_speed, COUNT(*) FROM transactions "
         f"WHERE {_y(2022).replace('t.', '')} GROUP BY shipping_speed"),
        ("same for 2023",
         "SELECT shipping_speed, COUNT(*) FROM transactions "
         f"WHERE {_y(2023).replace('t.', '')} GROUP BY shipping_speed"),
    ]),
    ("unrelated question after history ignores the old filters", [
        ("total revenue in germany in 2022", _rev(_DE22)),
        ("how many gold customers are in france",
         "SELECT COUNT(*) FROM customers WHERE country_code = 'FR' AND customer_tier = 'Gold'"),
    ]),
    ("follow-up with nothing to follow is not guessed at", [
        ("and last month?", ASK),
    ]),
    ("a no-rows turn is still something to follow", [
        ("total revenue in germany in 1999", _rev(f"c.country_code = 'DE' AND {_y(1999)}")),
        ("what about 2022", _rev(f"c.country_code = 'DE' AND {_y(2022)}")),
    ]),
]


# --------------------------------------------------------------------------

def _numbers(rows) -> list[tuple]:
    """Rows -> sorted tuples of their numeric cells (labels and names dropped)."""
    out = []
    for row in rows:
        values = row.values() if isinstance(row, dict) else row
        nums = []
        for v in values:
            if isinstance(v, bool) or v is None:
                nums.append(None if v is None else float(v))
            elif isinstance(v, (int, float)):
                nums.append(round(float(v), 2))
        out.append(tuple(nums))
    return sorted(out, key=lambda r: tuple(-1e18 if x is None else x for x in r))


def same_answer(got_rows, expected_rows) -> bool:
    return _numbers(got_rows) == _numbers(expected_rows)


def _reference_rows(sql: str):
    cur_rows = db.run_readonly_query(sql)
    return cur_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--min", type=float, default=0.0, help="exit 1 if accuracy is below this (0-1)")
    args = ap.parse_args()

    with pipeline._cache_lock:
        pipeline._cache.clear()          # an honest run: nothing served from cache

    total = passed = 0
    started = time.time()
    for ci, (title, steps) in enumerate(CONVERSATIONS, 1):
        sid = f"golden-{ci}-{int(started)}"
        print(f"\n[{ci}] {title}")
        for q, ref in steps:
            total += 1
            t0 = time.time()
            res = pipeline.answer_question(q, session_id=sid)
            ms = int((time.time() - t0) * 1000)
            if ref == ASK:
                ok = res.text == pipeline.NEEDS_CONTEXT_REPLY and res.sql is None
                detail = "" if ok else f"expected a request to restate, got: {res.text[:100]}"
            elif res.rows is None:
                ok, detail = False, f"no result: {res.text[:120]}"
            else:
                try:
                    expected = _reference_rows(ref)
                    ok = same_answer(res.rows, expected)
                    detail = "" if ok else f"got {_numbers(res.rows)[:4]}  expected {_numbers(expected)[:4]}"
                except Exception as exc:          # a broken reference is a test bug, say so
                    ok, detail = False, f"REFERENCE QUERY FAILED ({exc})"
            passed += ok
            print(f"  {'PASS' if ok else 'FAIL'}  {ms:>6} ms  {q}")
            if not ok or args.verbose:
                if detail:
                    print(f"         {detail}")
                if res.interpreted_as:
                    print(f"         read as: {res.interpreted_as}")
                if res.sql:
                    print(f"         sql: {res.sql}")
        memory.clear(sid)

    acc = passed / total if total else 0.0
    print(f"\nFollow-up accuracy: {passed}/{total} = {acc:.0%}  ({time.time() - started:.0f}s)")
    return 0 if acc >= args.min else 1


if __name__ == "__main__":
    raise SystemExit(main())
