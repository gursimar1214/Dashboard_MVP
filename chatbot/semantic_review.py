"""Model-assisted semantic review for generated SQL.

The SQL validator protects the database; this reviewer checks whether a
read-only query appears to answer the user's request. It is intentionally
general-purpose and uses the same schema and metric definitions as generation.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from . import llm
from .schema_glossary import COMMON_MISTAKES, METRICS_GLOSSARY, SCHEMA_DESCRIPTION, VALID_VALUES_TEXT


class SQLReview(BaseModel):
    matches_request: bool = Field(
        ..., description="True only if the SQL appears to answer the full user request."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete mismatches between the request and SQL; empty when it matches.",
    )


_REVIEW_SYSTEM = f"""
You are a cautious reviewer of read-only SQLite queries for an ecommerce dashboard.
Compare the user's complete request with the candidate SQL. Do not rewrite the SQL.

Schema:
{SCHEMA_DESCRIPTION}

Valid values:
{VALID_VALUES_TEXT}

Metric definitions:
{METRICS_GLOSSARY}

Data-grain and join rules:
{COMMON_MISTAKES}

Check that the SQL includes every requested metric, output field, filter, and grouping;
uses the correct metric definition and table grain; and implements ranking scope correctly.
Use this review sequence before setting matches_request=true:
1. List the requested output dimensions separately from filter values. A multi-value
   filter (for example country_code IN ('IN','US')) is not a substitute for returning
   each requested country separately. If the request compares payment methods by
   country, both country and payment method must survive into the final output and
   aggregation grain.
2. For each metric, identify its source table and natural grain. Trace row counts and
   duplication through every JOIN and CTE. A query can be valid SQLite and still be
   semantically wrong because a one-to-many JOIN duplicates a measure.
3. For campaign budgets plus session counts, budgets must be summed once per campaign;
   pre-aggregate sessions by campaign before joining. COUNT(DISTINCT session_id) only
   repairs that count and does not repair a multiplied SUM(campaign_budget).
4. For a customer-level metric grouped by a session attribute such as device_type,
   first deduplicate to one row per customer and device. Joining customers directly to
   raw sessions produces session-weighted averages and is not a customer-level average.
5. Verify the final SELECT, GROUP BY and window partitions express every requested
   output dimension. A WHERE IN clause alone never creates separate grouped results.

In particular, "top N per/for/within each group" requires an independent rank inside each
group (for example ROW_NUMBER() OVER (PARTITION BY group_column ...)) and a filter on that
rank. A global ORDER BY ... LIMIT N does not return N rows per group. A global LIMIT that
truncates one group before another is also a mismatch. Do not confuse a filter for multiple
values with a request for separate top-N lists. A valid, preferred pattern is to aggregate
the requested metric to one row per group and entity in a CTE, rank those rows in a second
CTE with ROW_NUMBER() OVER (PARTITION BY group_column ORDER BY metric DESC), and filter
the final result to rank <= N. This pattern does not need a separate UNION query per group.
When that partitioned rank and rank filter are present, treat the per-group ranking
requirement as satisfied; do not claim that a global limit is required. For aggregated
metrics, ranking an aggregate alias from a prior CTE is valid and avoids relying on
dialect-specific aggregate/window ordering behavior. Inspect all CTEs and the final SELECT
before reporting that a requested filter, rank, group, or metric is missing. `customer_id`
is unique in `customers`, so a customer's tier is functionally determined by that ID.

Set matches_request=false only for a concrete semantic mismatch. Do not reject a query for
formatting, alias style, or equivalent SQL formulations. When uncertain about a material
requirement, report the uncertainty as an issue rather than approving it.
""".strip()


def review_sql(request: str, sql: str) -> SQLReview:
    """Review generated SQL against the original request and known data semantics."""
    return llm.generate_structured(
        system=_REVIEW_SYSTEM,
        prompt=f"User request:\n{request}\n\nCandidate SQL:\n{sql}",
        schema=SQLReview,
        max_tokens=220,
        name="sql-semantic-review",
    )
