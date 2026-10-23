"""Resolve relative periods against the latest date present in the data.

The model's interpretation of phrases such as "this year" is not reliable.
These helpers turn supported phrases into half-open ISO date ranges without
using the machine's wall clock.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date  # exclusive
    label: str

    def sql_predicate(self, column: str = "transaction_timestamp") -> str:
        return f"{column} >= '{self.start.isoformat()}' AND {column} < '{self.end.isoformat()}'"

    def instruction(self, column: str = "transaction_timestamp") -> str:
        return f"{self.label} means {self.sql_predicate(column)} (end date is exclusive)."


def resolve_time_window(value: Optional[str], reference_date: str) -> Optional[DateWindow]:
    """Resolve a recognized time phrase. Unsupported/ambiguous phrases return None."""
    if not value:
        return None
    try:
        ref = date.fromisoformat(reference_date[:10])
    except (TypeError, ValueError):
        return None

    phrase = " ".join(value.lower().strip().split())
    day_after = ref + timedelta(days=1)

    # Explicit calendar periods are deterministic too.
    if re.fullmatch(r"20\d{2}", phrase):
        year = int(phrase)
        return DateWindow(date(year, 1, 1), date(year + 1, 1, 1), phrase)
    match = re.fullmatch(r"([a-z]+) (20\d{2})", phrase)
    if match:
        month = _month_number(match.group(1))
        if month:
            year = int(match.group(2))
            start = date(year, month, 1)
            return DateWindow(start, _next_month(start), phrase)
    match = re.fullmatch(r"q([1-4])(?: (20\d{2}))?", phrase)
    if match and match.group(2):
        quarter = int(match.group(1))
        year = int(match.group(2))
        start = date(year, (quarter - 1) * 3 + 1, 1)
        return DateWindow(start, _add_months(start, 3), phrase)

    if phrase in {"this year", "current year", "ytd", "year to date"}:
        return DateWindow(date(ref.year, 1, 1), day_after, phrase)
    if phrase == "last year":
        return DateWindow(date(ref.year - 1, 1, 1), date(ref.year, 1, 1), phrase)
    if phrase in {"this month", "current month"}:
        return DateWindow(date(ref.year, ref.month, 1), day_after, phrase)
    if phrase in {"last month", "previous month"}:
        end = date(ref.year, ref.month, 1)
        return DateWindow(_previous_month(end), end, phrase)
    if phrase in {"this quarter", "current quarter"}:
        month = ((ref.month - 1) // 3) * 3 + 1
        return DateWindow(date(ref.year, month, 1), day_after, phrase)
    if phrase in {"last quarter", "previous quarter"}:
        month = ((ref.month - 1) // 3) * 3 + 1
        end = date(ref.year, month, 1)
        return DateWindow(_add_months(end, -3), end, phrase)
    if phrase in {"this week", "current week"}:
        start = ref - timedelta(days=ref.weekday())
        return DateWindow(start, day_after, phrase)
    if phrase in {"last week", "previous week"}:
        this_week = ref - timedelta(days=ref.weekday())
        return DateWindow(this_week - timedelta(days=7), this_week, phrase)
    if phrase == "today":
        return DateWindow(ref, day_after, phrase)
    if phrase == "yesterday":
        return DateWindow(ref - timedelta(days=1), ref, phrase)

    match = re.fullmatch(r"(?:last|past|previous) (\d+) days?", phrase)
    if match:
        days = int(match.group(1))
        if 1 <= days <= 3660:
            return DateWindow(ref - timedelta(days=days - 1), day_after, phrase)
    return None


def _month_number(name: str) -> Optional[int]:
    for number in range(1, 13):
        if calendar.month_name[number].lower() == name or calendar.month_abbr[number].lower() == name:
            return number
    return None


def _next_month(start: date) -> date:
    return _add_months(start, 1)


def _previous_month(start: date) -> date:
    return _add_months(start, -1)


def _add_months(start: date, count: int) -> date:
    index = start.year * 12 + start.month - 1 + count
    year, month0 = divmod(index, 12)
    return date(year, month0 + 1, 1)
