"""Generate and review a safe, smaller answer after the main SQL path fails."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from . import llm
from .schema_glossary import COMMON_MISTAKES, METRICS_GLOSSARY, SCHEMA_DESCRIPTION, VALID_VALUES_TEXT


class PartialPlan(BaseModel):
    can_answer: bool = Field(..., description="Whether a useful requested portion can be answered safely.")
    answerable_part: Optional[str] = Field(
        default=None,
        description="A short, self-contained description of exactly what the SQL will answer.",
    )
    is_partial: bool = Field(..., description="True when the plan omits any part of the user's request.")
    omitted_parts: list[str] = Field(
        default_factory=list,
        description="Requested parts this fallback will not answer; empty if the full request is answered.",
    )
    sql: Optional[str] = Field(
        default=None,
        description="One read-only SQLite SELECT, possibly WITH-wrapped; null if no safe answer is possible.",
    )


class PartialReview(BaseModel):
    approved: bool = Field(..., description="Whether the fallback is a faithful, safe subset of the request.")
    issues: list[str] = Field(default_factory=list, description="Concrete reasons to reject the fallback.")


_PLAN_SYSTEM = f"""
You are a fallback query planner for an ecommerce dashboard. The main query failed
semantic review, SQL validation, or execution. Find one simpler useful answer that
is explicitly requested and can be answered from the schema. It may answer the full
request if a simpler query can do so; otherwise answer only a clear subset.

Schema:
{SCHEMA_DESCRIPTION}

Allowed values:
{VALID_VALUES_TEXT}

Metric definitions:
{METRICS_GLOSSARY}

Query/grain rules:
{COMMON_MISTAKES}

Rules:
- Never invent filters, metrics, periods, or ranking meanings. If "top" has no clear
  metric and context does not resolve it, set can_answer=false and sql=null.
- Preserve explicit filters that apply to the portion being answered.
- If a request has independent groups or components and only one can be answered
  confidently, choose a useful requested component and name every omitted component.
- Do not silently turn a per-group ranking into a global ranking. A partial answer
  may cover one explicitly requested group, but must name the other groups as omitted.
- Use a simple read-only SQLite SELECT or WITH-wrapped SELECT. Do not use UNION.
- The SQL must answer exactly answerable_part. Do not include extra unrelated data.
- If no useful, unambiguous subset exists, set can_answer=false, answerable_part=null,
  and sql=null. Return JSON matching the supplied schema.
""".strip()


_REVIEW_SYSTEM = f"""
Review a proposed fallback for an ecommerce question. Decide whether its scope is
faithfully contained in the original request and whether its SQL answers that scope.

Schema:
{SCHEMA_DESCRIPTION}

Allowed values:
{VALID_VALUES_TEXT}

Metric definitions:
{METRICS_GLOSSARY}

Query/grain rules:
{COMMON_MISTAKES}

Approve only if the answerable part is requested or is a genuine component of the
request, the omitted parts accurately describe what is left unanswered, no unsupported
assumption was added, and the SQL matches the answerable part. For a partial ranking,
verify that SQL is limited to the named group(s) and does not claim to rank all groups.
Reject if the metric or ranking basis is ambiguous and the plan guessed one. Do not
reject equivalent valid SQLite forms or request additional optional columns.
Return JSON matching the supplied schema.
""".strip()


def propose(question: str, failure_reason: str, failed_sql: str = "") -> PartialPlan:
    prompt = (
        f"Original user request:\n{question}\n\n"
        f"Why the previous query failed:\n{failure_reason}\n\n"
        f"Previous SQL (untrusted; do not execute or copy blindly):\n{failed_sql}"
    )
    return llm.generate_structured(
        system=_PLAN_SYSTEM,
        prompt=prompt,
        schema=PartialPlan,
        max_tokens=420,
        name="partial-query-plan",
    )


def review(question: str, plan: PartialPlan) -> PartialReview:
    prompt = (
        f"Original user request:\n{question}\n\n"
        f"Proposed answerable part:\n{plan.answerable_part}\n\n"
        f"Is partial: {plan.is_partial}\n"
        f"Omitted parts: {plan.omitted_parts}\n\n"
        f"Candidate SQL:\n{plan.sql}"
    )
    return llm.generate_structured(
        system=_REVIEW_SYSTEM,
        prompt=prompt,
        schema=PartialReview,
        max_tokens=220,
        name="partial-query-review",
    )
