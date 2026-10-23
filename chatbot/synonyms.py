"""
synonyms.py -- the synonym layer: turns loose wording into the exact terms
this database uses, BEFORE the model sees the question. Pure Python, no
model, no third-party imports, a fraction of a millisecond per question.

What it does for one question
-----------------------------
1. Fixes typos of domain words ("emal opning rate" -> "email opening rate")
   -- narrowly: first letter must match, the word must be a near-miss of a
   known schema word, and everyday words are protected.
2. Recognises, in canonical form:
     metric     "opens the emails" / "email engagement" -> email_open_rate
     dimension  "which country ..." / "by device"       -> country / device
     filters    "germany", "gold", "mobile", "paypal"   -> country_code=DE ...
     time       "2023", "March 2023", "last month"
     rank       "highest" / "lowest" / "top 5"
3. Produces short disambiguation HINTS for the model where the schema is
   genuinely ambiguous ("email" is a column, a traffic source AND a campaign
   type; "region" is two different columns).
4. Counts LEFTOVER content words it did not understand. slotmem.py only
   trusts the analysis (and skips the model's history reading) when this is
   zero -- precision over recall: anything unfamiliar goes to the model as
   before.

The vocabulary mirrors schema_glossary.py / value_fix.py; keep them in sync
when a column or a valid value is added (tests/test_synonyms.py checks the
valid values).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Typo correction (narrow on purpose)
# --------------------------------------------------------------------------

_TYPO_VOCAB = sorted({
    "email", "emails", "revenue", "customers", "customer", "country", "countries",
    "region", "regions", "loyalty", "loyal", "review", "reviews", "rating", "ratings",
    "credit", "session", "sessions", "bounce", "campaign", "campaigns", "budget",
    "discount", "payment", "shipping", "desktop", "mobile", "tablet", "organic",
    "social", "income", "urban", "internet", "penetration", "average", "platinum",
    "bronze", "silver", "germany", "france", "india", "overnight", "express",
    "standard", "orders", "order", "opening", "opens", "engagement", "device",
    "devices", "source", "sources", "channel", "channels", "month", "monthly",
    "yearly", "highest", "lowest", "paypal", "crypto", "affiliate", "display",
    "search", "traffic", "transactions", "purchases", "duration", "pages",
    "tier", "tiers", "aov", "value", "limit", "score", "rate", "gold",
})
_TYPO_VOCAB_SET = frozenset(_TYPO_VOCAB)

# Everyday words that sit close to a vocabulary word but are NOT typos.
_TYPO_PROTECT = frozenset({
    "table", "tables", "research", "brand", "bonus", "spend", "spent", "spending",
    "sale", "valued", "values", "ratio", "ratios", "rated", "limits", "scored",
    "scores", "paid", "pays", "pager", "mail", "gift", "golden", "emailed",
    "ordered", "ordering", "searched", "searches", "displays", "displayed",
    "standards", "sourced", "tiered", "months", "years", "average", "averages",
    "open", "opened", "opener", "opens", "orders", "ordering", "paying", "payments",
    "region", "regions", "tier", "tiers", "session", "sessions", "discounts",
})

_WORD = re.compile(r"[A-Za-z]+")


def fix_typos(text: str) -> tuple[str, dict[str, str]]:
    """Correct near-miss spellings of schema words. Returns (text, {typo: fix})."""
    fixed: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        word = m.group(0)
        low = word.lower()
        if len(low) < 4 or low in _TYPO_VOCAB_SET or low in _TYPO_PROTECT or low in STOP:
            return word
        for cand in difflib.get_close_matches(low, _TYPO_VOCAB, n=3, cutoff=0.84):
            if cand[0] == low[0] and abs(len(cand) - len(low)) <= 2:
                fixed[low] = cand
                return cand
        return word

    return _WORD.sub(repl, text), fixed


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

STOP = frozenset("""
a an the and or also what whats what's about how show me tell give list get is are was were be been
do does did of for in on at to by per with only just from over that this these those my our
your please can could would will you i we us it its there their they them have has had total
overall number numbers count counts rate rates score scores value values average avg mean
amount customers customer users user data stats figures figure many much most least all any
every each across where when which who whom now currently current so far yet still ever then
but if as into out up down new same instead other others else one ones thanks thank traffic
visitors visits metric metrics us s
""".split())

# metric key -> (label used in the rewritten question, regex). ORDER MATTERS:
# the first pattern to match a phrase "uses it up", so specific phrases come
# before the general ones they contain ("average order value" before "order").
_M = re.IGNORECASE
_METRICS: list[tuple[str, str, re.Pattern]] = [
    ("aov", "average order value",
     re.compile(r"\baov\b|\baverage\s+(?:order|basket)\s+(?:value|size|amount)\b|\bavg\.?\s+order\s+value\b|\bbasket\s+size\b", _M)),
    ("high_value_share", "high-value order share",
     re.compile(r"\bhigh[- ]value\s+(?:order\s+)?(?:share|rate|percentage|orders?)\b", _M)),
    ("items_per_order", "average items per order",
     re.compile(r"\bitems?\s+per\s+order\b|\baverage\s+items?\b|\bitems?\s+count\b", _M)),
    ("email_open_rate", "average email open rate",
     re.compile(
         r"\be-?mails?\s+(?:open\w*|engag\w*)(?:\s+rates?)?\b"
         r"|\bopen(?:s|ed|ing)?\s+(?:the\s+|their\s+|our\s+|those\s+)?e-?mails?\b"
         r"|\b(?:e-?mail\s+)?open(?:ing)?\s+rates?\b|\be-?mail\s+opens?\b", _M)),
    ("session_duration", "average session duration",
     re.compile(r"\bsession\s+(?:duration|length|time)\b|\btime\s+(?:on\s+site|spent)\b|\bhow\s+long\b", _M)),
    ("pages", "average pages per session",
     re.compile(r"\bpages?\s+(?:viewed|per\s+session|views?)\b|\bpage\s*views?\b|\bpages\b", _M)),
    ("cart_additions", "average cart additions per session",
     re.compile(r"\bcart\s+additions?\b|\badd(?:ed|s)?\s+to\s+(?:the\s+)?cart\b|\bcarts?\b", _M)),
    ("discount", "discount usage rate",
     re.compile(r"\bdiscount(?:\s+usage|\s+use|\s+rate|s|ed)?(?:\s+rate)?\b", _M)),
    ("campaign_budget", "total campaign budget",
     re.compile(r"\b(?:campaign\s+)?budgets?\b|\b(?:ad|marketing|campaign)\s+spend\b", _M)),
    ("campaigns_count", "number of campaigns",
     re.compile(r"\bhow\s+many\s+campaigns?\b|\bnumber\s+of\s+campaigns?\b|\bcampaigns?\s+count\b", _M)),
    ("customers_count", "number of customers",
     re.compile(r"\bhow\s+many\s+customers?\b|\bnumber\s+of\s+customers?\b|\bcustomers?\s+count\b|\bcustomer\s+base\b", _M)),
    ("credit_limit", "average credit limit",
     re.compile(r"\bcredit\s+limits?\b", _M)),
    ("loyalty", "average loyalty score",
     re.compile(r"\bloyalty(?:\s+scores?)?\b|\bhow\s+loyal\b|\bloyal\b", _M)),
    ("review_score", "average review score",
     re.compile(r"\b(?:avg\.?\s+|average\s+)?(?:review|rating)s?(?:\s+scores?)?\b|\bsatisfaction\b", _M)),
    ("income", "average income",
     re.compile(r"\b(?:average\s+|avg\.?\s+)?income\b|\bearnings\b", _M)),
    ("urban_ratio", "average urban ratio",
     re.compile(r"\burban(?:ization|isation)?(?:\s+(?:ratio|share|rate))?\b", _M)),
    ("internet_penetration", "average internet penetration",
     re.compile(r"\binternet(?:\s+(?:penetration|access|usage|coverage))?\b|\bpenetration\b", _M)),
    ("age", "average customer age",
     re.compile(r"\b(?:average|avg\.?|mean)\s+age\b|\bhow\s+old\b|(?<!\bby\s)(?<!\bper\s)\bage\b(?!\s+groups?)", _M)),
    ("bounce_rate", "bounce rate",
     re.compile(r"\bbounce(?:\s+rate)?\b|\bbounced\b|\bbounces\b", _M)),
    ("sessions", "number of sessions",
     re.compile(r"\bsessions?\b|\bvisits?\b", _M)),
    ("revenue", "revenue",
     re.compile(r"\brevenues?\b|\bturnover\b|\bamount\s+of\s+money\b", _M)),
    ("orders", "number of orders",
     re.compile(r"\borders?\b|\bsales\b|\bpurchases?\b|\btransactions?\b|\bbought\b", _M)),
]
_METRIC_LABEL = {k: label for k, label, _ in _METRICS}

# Which table family a metric lives on -> which "region" column is meant.
_METRIC_GROUP = {
    "email_open_rate": "customer", "loyalty": "customer", "review_score": "customer",
    "credit_limit": "customer", "age": "customer", "customers_count": "customer",
    "revenue": "order", "orders": "order", "aov": "order", "discount": "order",
    "items_per_order": "order", "high_value_share": "order",
    "sessions": "session", "bounce_rate": "session", "session_duration": "session",
    "pages": "session", "cart_additions": "session",
    "income": "geo", "urban_ratio": "geo", "internet_penetration": "geo",
    "campaign_budget": "campaign", "campaigns_count": "campaign",
}
_REGION_COLUMN = {
    "customer": "customers.region (the customer's home region)",
    "order": "customers.region (JOIN customers; the customer's home region)",
    "session": "sessions.geo_ip_region",
    "geo": "geo_data.geo_ip_region",
    "campaign": "marketing_campaigns.region_target",
}

# dimension key -> (label for the rewritten question, noun regex)
_DIMENSIONS: list[tuple[str, str, str]] = [
    ("region_tier", "region tier", r"region\s+tiers?"),
    ("region", "region", r"regions?"),
    ("country", "country", r"countr(?:y|ies)|nations?"),
    ("tier", "customer tier", r"tiers?|membership\s+levels?|segments?"),
    ("payment_method", "payment method", r"payment(?:\s+(?:methods?|types?|modes?))?"),
    ("shipping_speed", "shipping speed", r"shipping(?:\s+(?:speeds?|methods?|types?))?|delivery(?:\s+speeds?)?"),
    ("device", "device type", r"devices?(?:\s+types?)?"),
    ("traffic_source", "traffic source", r"traffic\s+sources?|sources?|channels?"),
    ("campaign_type", "campaign type", r"campaign(?:\s+types?)?|campaigns\s+types?"),
]
_DIM_LABEL = {k: label for k, label, _ in _DIMENSIONS}
_DIM_PATTERNS: list[tuple[str, re.Pattern]] = [
    (k, re.compile(
        rf"\b(?:by|per|across|each|every|which|what|whichever|split\s+by|grouped\s+by|broken\s+down\s+by|breakdown\s+by)"
        rf"\s+(?:the\s+)?(?:customer\s+)?(?:{noun})\b|\b(?:{noun})[- ]?wise\b", _M))
    for k, _, noun in _DIMENSIONS
]
_TOP_N_DIM: list[tuple[str, re.Pattern]] = [
    (k, re.compile(rf"\b(top|bottom)\s+(\d{{1,2}})\s+(?:{noun})\b", _M))
    for k, _, noun in _DIMENSIONS
]
_MONTH_DIM = re.compile(r"\b(?:monthly|month\s+over\s+month|(?:by|per|each|every)\s+month)\b", _M)
_YEAR_DIM = re.compile(r"\b(?:yearly|annually|year\s+over\s+year|(?:by|per|each|every)\s+year)\b", _M)

# Filters ------------------------------------------------------------------
_COUNTRY_CI = {
    "US": re.compile(r"\busa\b|\bu\.s\.a?\.?|\bunited\s+states(?:\s+of\s+america)?\b|\bamerica(?:n)?\b", _M),
    "UK": re.compile(r"\buk\b|\bu\.k\.|\bunited\s+kingdom\b|\bgreat\s+britain\b|\bbritain\b|\bbritish\b|\bengland\b", _M),
    "DE": re.compile(r"\bgermany\b|\bgerman\b|\bdeutschland\b", _M),
    "FR": re.compile(r"\bfrance\b|\bfrench\b", _M),
    "IN": re.compile(r"\bindia\b|\bindian\b", _M),
}
# Two-letter codes only count when typed in capitals ("US", not "us").
_COUNTRY_CS = {c: re.compile(rf"\b{c}\b") for c in ("US", "UK", "DE", "FR")}

_TIER = re.compile(r"\b(bronze|silver|gold|platinum)\b", _M)
_PAYMENT = re.compile(r"\b(debit|credit)[ _-]?cards?\b|\b(paypal|crypto)(?:currency)?\b", _M)
_SHIPPING = re.compile(r"\b(standard|express|overnight)\b(?:\s+(?:shipping|delivery))?", _M)
_DEVICE = re.compile(r"\b(desktop|mobile|tablet)s?\b", _M)
_CAMPAIGN_TYPE = re.compile(r"\b(affiliate|search|display|social|e-?mail)(?=\s+campaigns?\b)|\b(affiliate|display)\b", _M)
_TRAFFIC = re.compile(
    r"\b(organic|ads?|social|e-?mail)\s+(?:traffic|sources?|channels?)\b"
    r"|\b(?:from|via|through)\s+(organic|ads?|social|e-?mail)\b(?!\s+(?:campaigns?|open\w*|engag\w*))", _M)

# Time -------------------------------------------------------------------------
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_TIME_MONTH_YEAR = re.compile(rf"\b({_MONTHS})\s+(20[12]\d)\b", _M)
_TIME_QUARTER = re.compile(r"\bq([1-4])(?:\s*(20[12]\d))?\b", _M)
_TIME_YEAR = re.compile(r"\b(20[12]\d)\b")
_TIME_RELATIVE = re.compile(
    r"\b(?:last|this|past|previous|current)\s+(?:month|year|quarter|week|\d+\s+(?:days?|weeks?|months?|years?))\b"
    r"|\byesterday\b|\btoday\b|\bytd\b|\byear\s+to\s+date\b", _M)
_TIME_MONTH_ONLY = re.compile(rf"\b(?:in|during|for)\s+({_MONTHS})\b", _M)

# Rank ---------------------------------------------------------------------------
_RANK_HIGH = re.compile(r"\b(?:highest|most|max(?:imum)?|best|top|largest|biggest|greatest|higher)\b", _M)
_RANK_LOW = re.compile(r"\b(?:lowest|least|min(?:imum)?|worst|bottom|smallest|fewest|lower)\b", _M)
_TOP_N = re.compile(r"\b(?:top|bottom|first|last)\s+(\d{1,2})\b", _M)

# "overall" style resets (follow-ups)
_CLEAR_COUNTRY = re.compile(r"\b(?:all|every|any)\s+countr(?:y|ies)\b|\bworldwide\b|\bglobally\b|\beverywhere\b", _M)
_CLEAR_TIER = re.compile(r"\b(?:all|every|any)\s+tiers?\b", _M)
_CLEAR_ALL = re.compile(r"\boverall\b|\bin\s+total\b|\bcombined\b|\baltogether\b|\bregardless\b|\bacross\s+all\b", _M)

# Phrases that make a question too intricate to rewrite from slots.
_COMPLEX = re.compile(
    r"\b(?:than|above|below|between|versus|vs|compar\w+|growth|trend|difference|"
    r"excluding|exclude|except|without|never|not|only\s+those|per\s+customer|per\s+order)\b", _M)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class Analysis:
    text: str                                   # typo-corrected question
    corrections: dict[str, str] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    dimension: Optional[str] = None
    filters: dict[str, str] = field(default_factory=dict)   # column -> "V" or "V1|V2"
    time: Optional[str] = None
    rank: Optional[str] = None                  # "highest" | "lowest"
    top_n: Optional[int] = None
    clear: set[str] = field(default_factory=set)            # column names, or "*"
    hints: list[str] = field(default_factory=list)
    leftover: int = 0                           # unexplained content words
    complex: bool = False                       # multi-metric / multi-time / "than"...
    consumed: list[str] = field(default_factory=list)

    @property
    def metric(self) -> Optional[str]:
        return self.metrics[0] if self.metrics else None

    @property
    def recognised_anything(self) -> bool:
        return bool(self.metrics or self.dimension or self.filters or self.time
                    or self.rank or self.clear)


class _Buf:
    """The question twice (original / lower-cased). Every match is blanked
    out, so a phrase can only be used once and what is left over is exactly
    the part nothing understood."""

    def __init__(self, text: str):
        self.orig, self.low = text, text.lower()
        self.consumed: list[str] = []

    def _blank(self, s: int, e: int) -> None:
        self.consumed.append(self.low[s:e].strip())
        pad = " " * (e - s)
        self.orig = self.orig[:s] + pad + self.orig[e:]
        self.low = self.low[:s] + pad + self.low[e:]

    def take(self, pat: re.Pattern, case_sensitive: bool = False) -> list[re.Match]:
        src = self.orig if case_sensitive else self.low
        found = list(pat.finditer(src))
        for m in found:
            self._blank(m.start(), m.end())
        return found

    def leftover_words(self) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9']+", self.low) if w not in STOP]


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:].lower()


def _add(filters: dict[str, str], col: str, value: str) -> None:
    cur = filters.get(col)
    if cur is None:
        filters[col] = value
    elif value not in cur.split("|"):
        filters[col] = "|".join(sorted(cur.split("|") + [value]))


# --------------------------------------------------------------------------
# analyze()
# --------------------------------------------------------------------------

def analyze(question: str) -> Analysis:
    """Typo-fix, then recognise metric / dimension / filters / time / rank."""
    text, corrections = fix_typos(" ".join((question or "").split()))
    a = Analysis(text=text, corrections=corrections)
    buf = _Buf(text)

    # ---- time first (so "2023" is not mistaken for a stray number) ----------
    times: list[str] = []
    for m in buf.take(_TIME_MONTH_YEAR):
        times.append(f"{_cap(m.group(1))} {m.group(2)}")
    for m in buf.take(_TIME_QUARTER):
        times.append(f"Q{m.group(1)}" + (f" {m.group(2)}" if m.group(2) else ""))
    for m in buf.take(_TIME_YEAR):
        times.append(m.group(1))
    for m in buf.take(_TIME_RELATIVE):
        times.append(" ".join(m.group(0).split()).lower())
    for m in buf.take(_TIME_MONTH_ONLY):
        times.append(_cap(m.group(1)))
    # a bare "in" / "during" left in front of a consumed time is harmless
    if len(set(times)) > 1:
        a.complex = True
    a.time = times[0] if times else None

    # ---- resets ("overall", "all countries") --------------------------------
    if buf.take(_CLEAR_COUNTRY):
        a.clear.add("country_code")
    if buf.take(_CLEAR_TIER):
        a.clear.add("customer_tier")
    if buf.take(_CLEAR_ALL):
        a.clear.add("*")

    # ---- campaign type first: it must not be eaten by "campaign budget" -----
    for m in buf.take(_CAMPAIGN_TYPE):
        v = (m.group(1) or m.group(2)).lower().replace("-", "")
        _add(a.filters, "campaign_type", _cap(v))

    # ---- "top 5 countries" = a dimension AND a count -------------------------
    dims: list[str] = []
    for key, pat in _TOP_N_DIM:
        for m in buf.take(pat):
            dims.append(key)
            a.top_n = int(m.group(2))
            a.rank = "lowest" if m.group(1).lower() == "bottom" else "highest"

    # ---- metrics --------------------------------------------------------------
    for key, _label, pat in _METRICS:
        if buf.take(pat) and key not in a.metrics:
            a.metrics.append(key)
    # "orders" is also matched inside "average order value" etc.; the buffer
    # blanking already prevents that, so >1 metric here is genuinely >1 ask.
    if len(a.metrics) > 1:
        a.complex = True

    # ---- dimension ------------------------------------------------------------
    for key, pat in _DIM_PATTERNS:
        if buf.take(pat):
            dims.append(key)
    if buf.take(_MONTH_DIM):
        dims.append("month")
    if buf.take(_YEAR_DIM):
        dims.append("year")
    if len(dims) > 1:
        a.complex = True            # "by country and tier" -> let the model do it
    a.dimension = dims[0] if dims else None

    # ---- filters --------------------------------------------------------------
    for code, pat in _COUNTRY_CI.items():
        if buf.take(pat):
            _add(a.filters, "country_code", code)
    for code, pat in _COUNTRY_CS.items():
        if buf.take(pat, case_sensitive=True):
            _add(a.filters, "country_code", code)
    for m in buf.take(_TIER):
        _add(a.filters, "customer_tier", _cap(m.group(1)))
    for m in buf.take(_PAYMENT):
        kind = (m.group(1) or m.group(2)).lower()
        _add(a.filters, "payment_method", {"debit": "debit_card", "credit": "credit_card"}.get(kind, kind))
    for m in buf.take(_TRAFFIC):
        v = (m.group(1) or m.group(2)).lower().replace("-", "")
        _add(a.filters, "traffic_source", "ads" if v == "ad" else v)
    for m in buf.take(_DEVICE):
        _add(a.filters, "device_type", m.group(1).lower())
    for m in buf.take(_SHIPPING):
        _add(a.filters, "shipping_speed", m.group(1).lower())

    # ---- rank -----------------------------------------------------------------
    for m in buf.take(_TOP_N):
        a.top_n = a.top_n or int(m.group(1))
        a.rank = "lowest" if m.group(0).lower().startswith(("bottom", "last")) else "highest"
    if buf.take(_RANK_HIGH):
        a.rank = a.rank or "highest"
    if buf.take(_RANK_LOW):
        a.rank = a.rank or "lowest"

    # ---- complexity + leftovers -----------------------------------------------
    if _COMPLEX.search(a.text):
        a.complex = True
    a.leftover = len(buf.leftover_words())
    a.consumed = [c for c in buf.consumed if c]

    a.hints = hints_for(a.metric, a.dimension, mentions_region=bool(re.search(r"\bregions?\b", a.text, _M)))
    return a


def hints_for(metric: Optional[str], dimension: Optional[str], mentions_region: bool = False) -> list[str]:
    """Short disambiguation notes for the model, only where the schema is
    genuinely ambiguous. Used for a typed question and for merged slots."""
    hints: list[str] = []
    if metric == "email_open_rate":
        hints.append(
            "email open rate = customers.email_open_rate (c.email_open_rate) from customers AS c "
            "(NOT traffic_source = 'email' and NOT campaign_type = 'Email')")
    if dimension == "region" or (mentions_region and not dimension):
        hints.append(
            f"'region' here means {_REGION_COLUMN[_METRIC_GROUP.get(metric or '', 'customer')]}; "
            "region codes look like 'region_7', they are not countries")
    return hints


def metric_label(key: str) -> str:
    return _METRIC_LABEL[key]


def dimension_label(key: str) -> str:
    if key == "month":
        return "month"
    if key == "year":
        return "year"
    return _DIM_LABEL[key]


def metric_group(key: Optional[str]) -> str:
    return _METRIC_GROUP.get(key or "", "customer")


def region_column_for(metric: Optional[str]) -> str:
    return _REGION_COLUMN[metric_group(metric)]
