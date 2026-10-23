"""
followup.py -- deterministic guard for follow-up questions that have nothing
to follow up on.

Problem: "and last month?" or "what about india" typed as the FIRST message
of a session (new tab, expired session, after a reset) has no metric. A
language model will still invent one -- usually "orders" or "revenue" -- run
it, show a confident number, and that made-up turn then becomes the memory
that later follow-ups build on. Asking the person to state the question in
full is always more accurate than guessing.

Pure Python, no model, no third-party imports.
"""
from __future__ import annotations

import re

# A message that STARTS like a follow-up.
_CUE = re.compile(
    r"^\s*(?:and|also|what\s+about|how\s+about|what\s+if|same|the\s+same|"
    r"instead|those|them|these|that\s+one)\b",
    re.IGNORECASE,
)

# Words that name a METRIC (something you can measure). If one is present the
# message says WHAT to measure, so it can be answered without history.
#
# Keep this to metric nouns only. Do NOT add words that are filter VALUES or
# GROUPING dimensions -- country, tier, device, shipping, campaign, a bare
# "email" (traffic source / campaign type) or a bare "credit" (credit_card):
# "what about email", "what about credit card", "also by device" and "same
# for gold customers" are real follow-ups, and treating those words as
# metrics would stop the guard from catching them.
# ("customers" is left out for the same reason.)
#
# tests/test_audit_fixes.py checks that every measurable column in
# schema_glossary.ALLOWED_TABLES has a phrase here, so a column added to the
# schema cannot silently fall out of this list again.
_METRIC = re.compile(
    r"\b(?:"
    # orders / revenue / sessions
    r"revenue|sales|orders?|aov|average\s+order|sessions?|bounce|"
    r"discount\w*|items?|purchases?|transactions?|spend|conversion|"
    # customer attributes
    r"loyalty|reviews?|ratings?|scores?|credit\s+limits?|"
    r"email\s+(?:open\w*|engagement)|open\s+rates?|opens|"
    # geo attributes
    r"income|urban\w*|internet|penetration|"
    # campaigns and session engagement
    r"budget|duration|pages?|carts?"
    r")\b",
    re.IGNORECASE,
)

# "age" is a metric ("average age by country") but also a grouping ("also by
# age"), so it only counts when it is not directly after "by".
_AGE_METRIC = re.compile(r"(?<!\bby\s)\bage\b", re.IGNORECASE)


def _names_a_metric(q: str) -> bool:
    q = " ".join(q.split())          # the "by age" lookbehind needs single spaces
    return bool(_METRIC.search(q) or _AGE_METRIC.search(q))



# Phrases that open a complete question of their own.
_FULL_QUESTION = re.compile(
    r"\b(?:how\s+many|how\s+much|which|list|show\s+me|top|best|worst|"
    r"highest|lowest|compare)\b",
    re.IGNORECASE,
)

MAX_FOLLOWUP_WORDS = 8     # longer than this is a real question, not a nudge
MAX_MODEL_FLAGGED_WORDS = 6


def looks_like_followup(question: str) -> bool:
    """True for short messages that begin like a follow-up and carry no
    metric of their own: "and last month?", "what about india", "same for
    gold customers", "also by device"."""
    q = (question or "").strip()
    if not q or len(q.split()) > MAX_FOLLOWUP_WORDS:
        return False
    if not _CUE.match(q):
        return False
    if _names_a_metric(q) or _FULL_QUESTION.search(q):
        return False
    return True


def needs_earlier_turn(question: str, has_context: bool, model_says_followup: bool = False) -> bool:
    """True when the message can only be understood with a previous turn
    and there is none.

    has_context          -- the session has remembered turns.
    model_says_followup  -- the model classified the message as "followup".
                            Trusted only for short messages: a long,
                            detailed question is answered even if the model
                            mislabelled it, so a good question is never
                            refused because of a wrong label.
    """
    if has_context:
        return False
    if looks_like_followup(question):
        return True
    return bool(model_says_followup and len((question or "").split()) <= MAX_MODEL_FLAGGED_WORDS)
