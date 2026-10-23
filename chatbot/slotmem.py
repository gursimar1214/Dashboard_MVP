"""
slotmem.py -- structured slot memory for follow-up questions.

The old way: paste the last few questions + SQL into the prompt as text and
let the model work out what changed. That costs prompt tokens (slow on an
iGPU) and leaves "is this a follow-up or a new question?" to the model.

The new way: remember the PARTS of the last question --

    metric: revenue | by: country | filters: customer_tier=Gold | time: 2022

-- and edit them in Python. "what about india" changes one filter, "by
device" changes the grouping, "overall" clears filters. The edited slots are
rendered back into one clear, self-contained question, so the model gets

    Question: revenue by country, where customer_tier = 'Gold', in 2022

with NO history block, and only has to write SQL (no standalone_question to
generate either). Fewer prompt tokens and fewer output tokens.

Three outcomes (Resolution.mode):

  fresh   The question names its own metric -> it is NEW. The old filters
          are dropped (never carried over silently) and the history is not
          sent. "orders by country" after "revenue in Germany for Gold"
          means orders by country, all customers.
  slots   A follow-up Python understood completely (nothing left over, no
          "than/above/between" complexity, a previous question to attach to)
          -> merged and rendered, history not sent.
  legacy  Anything else -> exactly the old behaviour: history text + the
          model decides. Precision over recall: when Python is not sure, the
          model sees what it always saw.

Pure Python, no third-party imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import followup, synonyms

_FILTER_ORDER = [
    "country_code", "customer_tier", "payment_method", "shipping_speed",
    "device_type", "traffic_source", "campaign_type",
]

# Words that point back at something said before ("revenue for them").
# A leading cue ("same for gold") is handled separately and is fine.
_REFERS_BACK = re.compile(
    r"\b(?:it|they|them|those|these|that|previous|earlier|above|before|again|too|"
    r"last\s+(?:answer|result|one)|the\s+same)\b", re.IGNORECASE)

_SAME_FOR_GROUP = re.compile(
    r"^(?:the\s+)?same(?:\s+for\s+(?:them|those|that\s+group|that\s+segment))?\s+"
    r"(?:by|per|across|grouped\s+by)\s+(?:the\s+)?"
    r"(?:country|countries|tier|tiers|device|devices|device\s+type|payment\s+method|"
    r"shipping\s+speed|traffic\s+source|campaign\s+type|region)\??$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------

@dataclass
class Slots:
    metric: Optional[str] = None
    dimension: Optional[str] = None
    filters: dict[str, str] = field(default_factory=dict)
    time: Optional[str] = None
    rank: Optional[str] = None
    top_n: Optional[int] = None

    def copy(self) -> "Slots":
        return Slots(self.metric, self.dimension, dict(self.filters), self.time, self.rank, self.top_n)

    def to_dict(self) -> dict:
        return {"metric": self.metric, "dimension": self.dimension, "filters": dict(self.filters),
                "time": self.time, "rank": self.rank, "top_n": self.top_n}

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional["Slots"]:
        if not d or not d.get("metric"):
            return None
        return Slots(d.get("metric"), d.get("dimension"), dict(d.get("filters") or {}),
                     d.get("time"), d.get("rank"), d.get("top_n"))

    # ---- render back into one self-contained question --------------------
    def render(self) -> str:
        metric = synonyms.metric_label(self.metric) if self.metric else "the metric"
        if self.dimension:
            dim = synonyms.dimension_label(self.dimension)
            if self.rank and self.top_n:
                head = f"{'top' if self.rank == 'highest' else 'bottom'} {self.top_n} {_plural(dim)} by {metric}"
            elif self.rank:
                head = f"which {dim} has the {self.rank} {metric}"
            else:
                head = f"{metric} by {dim}"
        else:
            head = metric
        parts = [head]
        conds = [_render_filter(c, self.filters[c]) for c in _FILTER_ORDER if c in self.filters]
        if conds:
            parts.append("where " + " and ".join(conds))
        if self.time:
            relative = re.match(r"(?:last|this|past|previous|current|yesterday|today|ytd|year to date)\b", self.time, re.I)
            parts.append(self.time if relative else f"in {self.time}")
        return ", ".join(parts)

    def hints(self) -> list[str]:
        return synonyms.hints_for(self.metric, self.dimension)


def _plural(label: str) -> str:
    return label[:-1] + "ies" if label.endswith("y") else label + "s"


def _render_filter(col: str, value: str) -> str:
    vals = value.split("|")
    if len(vals) == 1:
        return f"{col} = '{vals[0]}'"
    return f"{col} IN ({', '.join(repr(v) for v in vals)})"


def slots_from_analysis(a: synonyms.Analysis) -> Optional[Slots]:
    """Slots for a question Python understood completely, else None."""
    if not a.metric or a.leftover or a.complex:
        return None
    return Slots(a.metric, a.dimension, dict(a.filters), a.time, a.rank, a.top_n)


def slots_from_text(text: Optional[str]) -> Optional[Slots]:
    """Used after a model-answered turn: try to read its resolved question."""
    if not text:
        return None
    return slots_from_analysis(synonyms.analyze(text))


# --------------------------------------------------------------------------
# Merging a follow-up into the previous slots
# --------------------------------------------------------------------------

def merge(prev: Slots, a: synonyms.Analysis) -> Slots:
    s = prev.copy()
    if a.metric:                       # "and orders?" -> new metric, keep the rest
        s.metric = a.metric
    if "*" in a.clear:                 # "overall" -> no filters, no grouping
        s.filters.clear()
        s.dimension = None
        s.rank = s.top_n = None
    for col in a.clear:
        s.filters.pop(col, None)
    s.filters.update(a.filters)        # a new value for a kind REPLACES the old one
    if a.dimension:
        s.dimension = a.dimension
    if a.time:
        s.time = a.time
    if a.rank:
        s.rank = a.rank
        s.top_n = a.top_n
    elif a.top_n:
        s.top_n = a.top_n
    if not s.dimension:                # a ranking needs something to rank
        s.rank = s.top_n = None
    return s


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------

@dataclass
class Resolution:
    mode: str                              # "fresh" | "slots" | "legacy"
    model_question: str                    # what goes after "Question:" in the prompt
    reason: str                            # for logs / Langfuse
    hints: list[str] = field(default_factory=list)
    slots: Optional[Slots] = None          # to remember if this turn is answered
    resolved_text: Optional[str] = None    # shown to the user as "Understood as"
    analysis: Optional[synonyms.Analysis] = None

    @property
    def use_history(self) -> bool:
        return self.mode == "legacy"


def resolve(question: str, prev_slots: Optional[Slots], has_history: bool,
            analysis: Optional[synonyms.Analysis] = None) -> Resolution:
    a = analysis or synonyms.analyze(question)
    typed_fixed = a.text
    corrected = typed_fixed if a.corrections else None

    cue = followup._CUE.match(typed_fixed)
    body = typed_fixed[cue.end():] if cue else typed_fixed
    names_metric = bool(a.metric) or followup._names_a_metric(typed_fixed)
    refers_back = bool(_REFERS_BACK.search(body))
    bare_rank_edit = bool(re.fullmatch(r"(?:top|bottom|first|last)\s+\d{1,2}", typed_fixed.strip(), re.I))

    # A top-N request starts a new task when it is phrased as a full question,
    # even if the local synonym parser does not understand every noun in it.
    # This keeps phrases such as "top 10 customers in India with their tier"
    # from inheriting an unrelated metric from an earlier turn. Short edits
    # such as "top 10 instead" still begin with a follow-up cue and are merged.
    standalone_ranking = bool(
        a.top_n is not None and not cue and not refers_back and not bare_rank_edit
    )

    # 1) Nothing to follow: a plain question, typo-fixed.
    if not has_history:
        return Resolution("fresh", typed_fixed, "no history", a.hints,
                          slots_from_analysis(a), corrected, a)

    # 2) It names its own metric and does not point back: a NEW question.
    if (names_metric or standalone_ranking) and not cue and not refers_back:
        reason = "self-contained ranking; history dropped" if standalone_ranking and not names_metric else "names its own metric; history dropped"
        return Resolution("fresh", typed_fixed, reason,
                          a.hints, slots_from_analysis(a), corrected, a)

    # 3) A follow-up Python can fully understand.
    if prev_slots is None:
        return Resolution("legacy", typed_fixed, "follow-up but no slots from the last turn", a.hints, None, None, a)
    safe_same_group = bool(
        prev_slots and cue and _SAME_FOR_GROUP.fullmatch(typed_fixed.strip())
        and a.dimension and not a.metric and not a.filters and not a.time
        and not a.rank and not a.top_n and not a.clear and not a.leftover and not a.complex
        and synonyms.metric_group(prev_slots.metric) == "customer"
        and a.dimension in {"country", "tier", "device"}
    )
    if refers_back and not safe_same_group:
        return Resolution("legacy", typed_fixed, "points back with a pronoun", a.hints, None, None, a)
    if a.complex:
        return Resolution("legacy", typed_fixed, "complex wording (than/above/multi...)", a.hints, None, None, a)
    if a.leftover:
        return Resolution("legacy", typed_fixed, f"{a.leftover} word(s) not understood", a.hints, None, None, a)
    if not a.recognised_anything:
        return Resolution("legacy", typed_fixed, "nothing recognised", a.hints, None, None, a)
    if cue and names_metric and not a.metric:
        return Resolution("legacy", typed_fixed, "unknown metric word after a cue", a.hints, None, None, a)

    merged = merge(prev_slots, a)
    text = merged.render()
    return Resolution("slots", text, "merged into previous slots", merged.hints(), merged, text, a)
