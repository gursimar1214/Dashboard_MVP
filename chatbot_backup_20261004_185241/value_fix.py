"""
value_fix.py -- puts text values in generated SQL into the exact spelling
stored in the database, BEFORE the SQL is validated and run.

Why: SQLite compares text case-sensitively. If the model writes
customer_tier = 'platinum' but the data says 'Platinum', the query is
valid, runs, and silently returns 0. This module fixes that class of
mistake in code, so the answer no longer depends on the model spelling
things exactly right.

Column-aware (so 'email' stays 'email' for traffic_source but becomes
'Email' for campaign_type). Only touches values it knows; anything
unknown is left alone. Handles:
    col = 'x'   col != 'x'   col <> 'x'   col == 'x'
    col LIKE 'x'   col NOT LIKE 'x'
    col IN ('x','y')   col NOT IN ('x','y')
    UPPER(col) = 'X'   LOWER(col) = 'x'
Country names are mapped to codes (India -> 'IN').

Keep COLUMN_VALUES in sync with the database and with the "Valid values"
block in schema_glossary.py.
"""
from __future__ import annotations

import re

COLUMN_VALUES: dict[str, list[str]] = {
    "country_code": ["US", "UK", "DE", "FR", "IN"],
    "customer_tier": ["Bronze", "Silver", "Gold", "Platinum"],
    "traffic_source": ["organic", "ads", "social", "email"],
    "device_type": ["desktop", "mobile", "tablet"],
    "payment_method": ["debit_card", "credit_card", "paypal", "crypto"],
    "shipping_speed": ["standard", "express", "overnight"],
    "high_value_flag": ["Yes", "No"],
    "campaign_type": ["Affiliate", "Email", "Search", "Display", "Social"],
    "region_tier": ["Tier1", "Tier2", "Tier3"],
}

COUNTRY_ALIASES = {
    "india": "IN",
    "united states": "US", "united states of america": "US",
    "usa": "US", "america": "US",
    "united kingdom": "UK", "great britain": "UK", "britain": "UK",
    "england": "UK", "gb": "UK",
    "germany": "DE",
    "france": "FR",
}

_LOOKUP: dict[str, dict[str, str]] = {
    col: {v.lower(): v for v in values} for col, values in COLUMN_VALUES.items()
}
_LOOKUP["country_code"].update(COUNTRY_ALIASES)


def _canon(literal: str, lookup: dict[str, str]) -> str:
    key = literal.strip().lower()
    if key in lookup:
        return lookup[key]
    key2 = key.replace(" ", "_").replace("-", "_")
    return lookup.get(key2, literal)


# Column may be qualified: c.customer_tier, sessions.traffic_source
_COL = r"(?:(?:\b\w+)\s*\.\s*)?\b{col}\b"

_EQ = {
    col: re.compile(
        r"(?P<lhs>" + _COL.format(col=col) + r"\s*(?:==|=|!=|<>|LIKE|NOT\s+LIKE)\s*)"
        r"'(?P<val>[^']*)'",
        re.IGNORECASE,
    )
    for col in _LOOKUP
}
# Handle UPPER(col) = 'X' / LOWER(col) = 'x'
_FUNC_EQ = {
    col: re.compile(
        r"(?P<lhs>(?:UPPER|LOWER)\s*\(\s*" + _COL.format(col=col) + r"\s*\)\s*(?:=|==)\s*)"
        r"'(?P<val>[^']*)'",
        re.IGNORECASE,
    )
    for col in _LOOKUP
}
_IN = {
    col: re.compile(
        r"(?P<lhs>" + _COL.format(col=col) + r"\s+(?:NOT\s+)?IN\s*\()(?P<body>[^)]*)\)",
        re.IGNORECASE,
    )
    for col in _LOOKUP
}
_QUOTED = re.compile(r"'([^']*)'")


def canonicalize_values(sql: str) -> str:
    for col, lookup in _LOOKUP.items():
        # IN / NOT IN first, so we don't double-process the quoted values.
        sql = _IN[col].sub(
            lambda m, lk=lookup: m.group("lhs")
            + _QUOTED.sub(lambda q: f"'{_canon(q.group(1), lk)}'", m.group("body"))
            + ")",
            sql,
        )
        # UPPER(col) = 'x'  -- the case-insensitive pattern is fine to leave
        # the function in place; we just canonicalize the literal.
        sql = _FUNC_EQ[col].sub(
            lambda m, lk=lookup: f"{m.group('lhs')}'{_canon(m.group('val'), lk)}'",
            sql,
        )
        sql = _EQ[col].sub(
            lambda m, lk=lookup: f"{m.group('lhs')}'{_canon(m.group('val'), lk)}'",
            sql,
        )
    return sql