"""
fastpath.py -- answers the simplest, unambiguous KPI questions with a fixed
SQL query and NO model call (~0.1s instead of 5-10s on an iGPU).

Deliberately tiny and strict: a question must match one pattern exactly
(ignoring case, punctuation and a leading "what is / show me / total").
Anything with a filter, period, grouping or other wording does not match and
goes to the model as usual. Only used when the chat has no history.
"""
from __future__ import annotations

import re
from typing import Optional

from .date_resolver import resolve_time_window

_LEAD = re.compile(r"^(?:what(?:'s| is| are)|show me|tell me|give me|get|how many|how much)?\s*(?:the\s+)?(?:total\s+|overall\s+)?", re.I)

# normalised text -> SQL (written the same way the model is told to write it)
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Compare each customer's aggregated spend with the average aggregated
    # customer spend (not the average single order). Keep this narrowly
    # phrased so other interpretations still go through the model.
    (re.compile(r"^(?:(?:which|list|show(?: me)?) )?customers? (?:whose|with) (?:their )?total spend is (?:above|over|greater than) (?:the )?(?:overall )?average$"),
     "SELECT customer_id, SUM(order_value) AS total_spend FROM transactions GROUP BY customer_id HAVING SUM(order_value) > (SELECT AVG(customer_spend) FROM (SELECT SUM(order_value) AS customer_spend FROM transactions GROUP BY customer_id)) ORDER BY total_spend DESC"),
    (re.compile(r"^(?:how many |count )?customers? (?:have|whose) (?:a )?total spend (?:is )?(?:above|over|greater than) (?:the )?(?:overall )?average$"),
     "SELECT COUNT(*) AS customer_count FROM (SELECT customer_id, SUM(order_value) AS total_spend FROM transactions GROUP BY customer_id HAVING SUM(order_value) > (SELECT AVG(customer_spend) FROM (SELECT SUM(order_value) AS customer_spend FROM transactions GROUP BY customer_id)))"),
    (re.compile(r"^(?:number of )?customers?(?: (?:do we have|are there|in total))?$"),
     "SELECT COUNT(DISTINCT customer_id) AS customer_count FROM customers"),
    (re.compile(r"^(?:number of )?(?:orders|sales|transactions|purchases)(?: (?:do we have|are there|in total|did we have))?$"),
     "SELECT COUNT(*) AS orders FROM transactions"),
    (re.compile(r"^(?:number of )?sessions(?: (?:do we have|are there|in total|did we have))?$"),
     "SELECT COUNT(*) AS sessions FROM sessions"),
    (re.compile(r"^revenue$"),
     "SELECT SUM(order_value) AS revenue FROM transactions"),
    (re.compile(r"^(?:aov|average order value|avg order value)$"),
     "SELECT SUM(order_value) / COUNT(*) AS aov FROM transactions"),
    (re.compile(r"^bounce rate$"),
     "SELECT AVG(CASE WHEN bounce_flag THEN 1.0 ELSE 0.0 END) AS bounce_rate FROM sessions"),
    (re.compile(r"^revenue by payment method$"),
     "SELECT payment_method, SUM(order_value) AS revenue FROM transactions GROUP BY payment_method ORDER BY revenue DESC"),
    (re.compile(r"^(?:orders|sales|transactions) by payment method$"),
     "SELECT payment_method, COUNT(*) AS orders FROM transactions GROUP BY payment_method ORDER BY orders DESC"),
    (re.compile(r"^revenue by shipping speed$"),
     "SELECT shipping_speed, SUM(order_value) AS revenue FROM transactions GROUP BY shipping_speed ORDER BY revenue DESC"),
    (re.compile(r"^(?:orders|sales|transactions) by shipping speed$"),
     "SELECT shipping_speed, COUNT(*) AS orders FROM transactions GROUP BY shipping_speed ORDER BY orders DESC"),
    (re.compile(r"^sessions by device(?: type)?$"),
     "SELECT device_type, COUNT(*) AS sessions FROM sessions GROUP BY device_type ORDER BY sessions DESC"),
    (re.compile(r"^sessions by (?:traffic source|channel)$"),
     "SELECT traffic_source, COUNT(*) AS sessions FROM sessions GROUP BY traffic_source ORDER BY sessions DESC"),
]


def _normalise(q: str) -> str:
    q = re.sub(r"[^\w\s']", " ", (q or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    return q


def match(question: str) -> Optional[str]:
    """SQL for an exactly-matching simple question, else None."""
    q = _normalise(question)
    if not q:
        return None
    # try as typed, then with a leading "what is the total ..." stripped
    for candidate in (q, _LEAD.sub("", q, count=1).strip()):
        for pat, sql in _PATTERNS:
            if pat.match(candidate):
                return sql
    return None


_FILTER_COLUMNS = {"country_code", "customer_tier"}
_FILTER_VALUES = {
    "country_code": {"US", "UK", "DE", "FR", "IN"},
    "customer_tier": {"Bronze", "Silver", "Gold", "Platinum"},
}


def match_slots(
    metric: Optional[str],
    dimension: Optional[str],
    filters: Optional[dict[str, str]] = None,
    time: Optional[str] = None,
    rank: Optional[str] = None,
    top_n: Optional[int] = None,
    reference_date: Optional[str] = None,
) -> Optional[str]:
    """SQL templates for a few well-defined, frequently used metric shapes.

    This deliberately returns None for unsupported dimensions, filters, or
    periods so those questions continue through the model path.
    """
    filters = dict(filters or {})
    if not set(filters).issubset(_FILTER_COLUMNS):
        return None
    for column, raw in filters.items():
        if not set(raw.split("|")) <= _FILTER_VALUES[column]:
            return None

    window = resolve_time_window(time, reference_date or "") if time else None
    if time and window is None:
        return None

    if metric == "aov" and dimension is None:
        if rank or top_n:
            return None
        customer_filters = [
            (f"c.{col}" if col in {"country_code", "customer_tier"} else col, val)
            for col, val in filters.items()
        ]
        needs_customer = bool(customer_filters)
        from_sql = "transactions t JOIN customers c ON c.customer_id = t.customer_id" if needs_customer else "transactions t"
        conditions = [_filter_sql(col, val) for col, val in customer_filters]
        if window:
            conditions.append(window.sql_predicate("t.transaction_timestamp"))
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return f"SELECT SUM(t.order_value) / COUNT(*) AS aov FROM {from_sql}{where}"

    if metric != "email_open_rate" or dimension not in {None, "country", "tier", "device"} or time:
        return None
    if dimension is None and (rank or top_n):
        return None
    if not set(filters).issubset(_FILTER_COLUMNS):
        return None

    conditions = [_filter_sql(f"c.{col}", val) for col, val in filters.items()]
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    ordering = ""
    if rank in {"highest", "lowest"}:
        ordering = " ORDER BY avg_email_open_rate " + ("DESC" if rank == "highest" else "ASC")
    limit = ""
    if top_n:
        limit = f" LIMIT {max(1, min(int(top_n), 200))}"
    elif ordering:
        limit = " LIMIT 1"

    if dimension == "device":
        sql = (
            "WITH customer_devices AS (SELECT DISTINCT customer_id, device_type FROM sessions) "
            "SELECT d.device_type, AVG(c.email_open_rate) AS avg_email_open_rate "
            "FROM customer_devices d JOIN customers c ON c.customer_id = d.customer_id"
            f"{where} GROUP BY d.device_type{ordering}{limit}"
        )
        return sql

    group_col = {"country": "c.country_code", "tier": "c.customer_tier"}.get(dimension)
    if group_col:
        return (
            f"SELECT {group_col}, AVG(c.email_open_rate) AS avg_email_open_rate "
            f"FROM customers c{where} GROUP BY {group_col}{ordering}{limit}"
        )
    return f"SELECT AVG(c.email_open_rate) AS avg_email_open_rate FROM customers c{where}"


def _filter_sql(column: str, raw: str) -> str:
    values = raw.split("|")
    if len(values) == 1:
        return f"{column} = '{values[0]}'"
    return f"{column} IN ({', '.join(repr(v) for v in values)})"
