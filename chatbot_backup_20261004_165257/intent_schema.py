"""
intent_schema.py -- the JSON shape stage 2 is constrained to produce.

llm.py enforces this shape with grammar-constrained decoding (Ollama's
native `format` json-schema),
so the model cannot emit a field of the wrong type or a key outside this
schema. pipeline.py trusts the result completely.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    OVERVIEW = "overview"          # a single KPI: revenue, AOV, sessions, bounce rate, etc.
    TREND = "trend"                # a metric over time (monthly)
    BREAKDOWN = "breakdown"        # a metric grouped by a dimension
    LOOKUP = "lookup"              # rows for one specific customer_id
    RANKING = "ranking"            # top-N customers/rows by some metric
    COMPARISON = "comparison"      # two or more explicit values compared
    FOLLOWUP = "followup"          # only meaningful with conversation context
    OUT_OF_SCOPE = "out_of_scope"  # not answerable from these five tables


class ExtractedQuery(BaseModel):
    intent: Intent
    standalone_question: Optional[str] = Field(
        default=None,
        description="The question rewritten to be understandable with no conversation history "
                    "(metric, filters and period carried over from earlier turns spelled out).",
    )
    metric: Optional[str] = Field(
        default=None,
        description="One of: revenue, orders, aov, sessions, bounce_rate, "
                    "customers, discount_rate, high_value_share, items_count, "
                    "email_open_rate, loyalty_score, review_score, credit_limit, age, "
                    "session_duration, pages_viewed, cart_additions, campaign_budget, "
                    "average_income, urban_ratio, internet_penetration.",
    )
    group_by: Optional[str] = Field(
        default=None,
        description="One of: customer_tier, traffic_source, payment_method, "
                    "country_code, month, device_type, shipping_speed, campaign_type, "
                    "region (the customer's home region, customers.region), "
                    "geo_ip_region (where a session came from), region_tier.",
    )
    country: Optional[str] = Field(default=None, description="Country code or name mentioned.")
    tier: Optional[str] = Field(default=None, description="Customer tier mentioned.")
    channel: Optional[str] = Field(default=None, description="Traffic source / channel mentioned.")
    date_from: Optional[str] = Field(default=None, description="YYYY-MM-DD start, if a date range was mentioned.")
    date_to: Optional[str] = Field(default=None, description="YYYY-MM-DD end, if a date range was mentioned.")
    customer_id: Optional[int] = Field(default=None, description="Exact customer ID, if named.")
    top_n: Optional[int] = Field(default=None, description="N, for 'top N' / 'best N' questions.")
    order: Optional[str] = Field(default=None, description="'asc' or 'desc', for ranking questions.")
    comparison_values: Optional[list[str]] = Field(
        default=None, description="The explicit values being compared (e.g. ['Gold','Platinum'])."
    )
    followup_of_previous: bool = Field(
        default=False,
        description="True only if this question cannot be understood without the previous turn.",
    )