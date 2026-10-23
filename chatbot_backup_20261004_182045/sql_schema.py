"""
sql_schema.py -- the JSON shape stage 4 is constrained to produce.

Wrapping the SQL string inside a small JSON object (instead of asking
for bare SQL text) lets stage 4 use the same grammar-constrained
decoding mechanism as stage 2's intent extraction, via a JSON schema.

`rationale` is a short chain-of-thought field: asking the model to state
which metric/table it chose measurably improves SQL correctness for
ambiguous questions, and the field is ignored downstream.
"""
from pydantic import BaseModel, Field


class GeneratedSQL(BaseModel):
    sql: str = Field(
        ...,
        description="A single read-only SQLite SELECT statement, no trailing semicolon.",
    )
    rationale: str = Field(
        default="",
        description="One sentence: which metric and tables this query uses and why.",
    )