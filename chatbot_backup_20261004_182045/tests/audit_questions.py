"""
audit_questions.py -- live end-to-end check of the audit fixes.

Each case is a question asked as the FIRST message of a fresh chat, plus an
independent ground-truth SQL query that is run directly on the database. The
chatbot's answer is compared with the ground truth, so a wrong-but-confident
answer (wrong region column, NULLs sorted first, a campaign join the model
avoided...) fails instead of looking plausible.

    python -m chatbot.tests.audit_questions --dry     # run only the ground-truth SQL
    python -m chatbot.tests.audit_questions           # ask the chatbot (needs Ollama)
    python -m chatbot.tests.audit_questions -k email  # only cases whose id/question contains "email"

Questions are deliberately phrased differently from the few-shot examples in
pipeline.py. If you keep a golden list of your own, compare the two and rename
any question that appears in both, so the tests stay honest.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from typing import Any

from ..config import DB_PATH

# id, question, ground-truth SQL (+ optional alternative readings of an ambiguous question)
CASES: list[dict[str, Any]] = [
    # ---- previously undocumented columns, asked cold --------------------
    dict(id="email_avg", q="what is the average email open rate",
         gt="SELECT AVG(email_open_rate) FROM customers"),
    dict(id="email_region_top", q="which region has the highest email open rate",
         gt="SELECT region, AVG(email_open_rate) r FROM customers GROUP BY region ORDER BY r DESC LIMIT 1"),
    dict(id="email_lowest_null", q="which 3 customers have the lowest email open rate",
         gt="SELECT customer_id, email_open_rate FROM customers WHERE email_open_rate IS NOT NULL ORDER BY email_open_rate ASC LIMIT 3"),
    dict(id="loyalty_tier", q="average loyalty score by customer tier",
         gt="SELECT customer_tier, AVG(loyalty_score) FROM customers GROUP BY customer_tier"),
    dict(id="review_platinum", q="what is the average review score of platinum customers",
         gt="SELECT AVG(avg_review_score) FROM customers WHERE customer_tier = 'Platinum'"),
    dict(id="credit_country", q="average credit limit by country",
         gt="SELECT country_code, AVG(credit_limit) FROM customers GROUP BY country_code"),
    dict(id="age_india", q="what is the average age of customers in india",
         gt="SELECT AVG(age) FROM customers WHERE country_code = 'IN'"),
    dict(id="income_avg", q="what is the average income across all regions",
         gt="SELECT AVG(average_income) FROM geo_data"),
    dict(id="urban_tier", q="average urban ratio by region tier",
         gt="SELECT region_tier, AVG(urban_ratio) FROM geo_data GROUP BY region_tier"),
    dict(id="internet_tier_top", q="which region tier has the highest internet penetration",
         gt="SELECT region_tier, AVG(internet_penetration) p FROM geo_data GROUP BY region_tier ORDER BY p DESC LIMIT 1"),
    dict(id="budget_type", q="total campaign budget by campaign type",
         gt="SELECT campaign_type, SUM(campaign_budget) FROM marketing_campaigns GROUP BY campaign_type"),
    dict(id="campaign_count", q="how many campaigns are there for each campaign type",
         gt="SELECT campaign_type, COUNT(*) FROM marketing_campaigns GROUP BY campaign_type"),
    dict(id="pages_avg", q="average pages viewed per session",
         gt="SELECT AVG(pages_viewed) FROM sessions"),
    dict(id="duration_mobile", q="average session duration on mobile",
         gt="SELECT AVG(session_duration) FROM sessions WHERE device_type = 'mobile'"),
    dict(id="discount_default", q="what is the discount usage rate",
         gt="SELECT AVG(CASE WHEN discount_applied THEN 1.0 ELSE 0.0 END) FROM transactions"),

    # ---- campaign joins (audit bug #4) ----------------------------------
    dict(id="campaign_bounce_top", q="which campaign type has the highest bounce rate",
         gt="SELECT m.campaign_type, AVG(CASE WHEN s.bounce_flag THEN 1.0 ELSE 0.0 END) b FROM sessions s JOIN marketing_campaigns m ON m.campaign_id = s.campaign_id GROUP BY m.campaign_type ORDER BY b DESC LIMIT 1"),
    dict(id="email_campaign_sessions", q="how many sessions came from email campaigns",
         gt="SELECT COUNT(*) FROM sessions s JOIN marketing_campaigns m ON m.campaign_id = s.campaign_id WHERE m.campaign_type = 'Email'"),
    dict(id="email_traffic_sessions", q="how many sessions came from email traffic",
         gt="SELECT COUNT(*) FROM sessions WHERE traffic_source = 'email'"),

    # ---- region ambiguity (audit bug #5) --------------------------------
    dict(id="region_revenue_top", q="which region has the most revenue",
         gt="SELECT c.region, SUM(t.order_value) r FROM transactions t JOIN customers c ON c.customer_id = t.customer_id GROUP BY c.region ORDER BY r DESC LIMIT 1"),
    dict(id="geoip_bounce_top3", q="top 3 geo ip regions by bounce rate",
         gt="SELECT geo_ip_region, AVG(CASE WHEN bounce_flag THEN 1.0 ELSE 0.0 END) b FROM sessions GROUP BY geo_ip_region ORDER BY b DESC LIMIT 3"),

    # ---- derived tables / subqueries (audit bug #3) ---------------------
    dict(id="above_avg_spend", q="how many customers have a total spend above the average customer spend",
         gt="SELECT COUNT(*) FROM (SELECT customer_id, SUM(order_value) s FROM transactions GROUP BY customer_id) WHERE s > (SELECT AVG(s2) FROM (SELECT SUM(order_value) s2 FROM transactions GROUP BY customer_id))",
         alts=["SELECT COUNT(*) FROM (SELECT customer_id, SUM(order_value) s FROM transactions GROUP BY customer_id) WHERE s > (SELECT SUM(order_value) * 1.0 / (SELECT COUNT(*) FROM customers) FROM transactions)"]),
    dict(id="top3_per_tier", q="top 3 customers by revenue in each customer tier",
         gt="SELECT customer_tier, customer_id, revenue FROM (SELECT c.customer_tier, c.customer_id, SUM(t.order_value) revenue, ROW_NUMBER() OVER (PARTITION BY c.customer_tier ORDER BY SUM(t.order_value) DESC) rn FROM customers c JOIN transactions t ON t.customer_id = c.customer_id GROUP BY c.customer_tier, c.customer_id) WHERE rn <= 3"),

    # ---- follow-up-looking but standalone, as a first message (audit bug #2)
    dict(id="fresh_what_about_email_open", q="what about email open rate",
         gt="SELECT AVG(email_open_rate) FROM customers"),
    dict(id="fresh_also_credit_by_tier", q="also credit limit by tier",
         gt="SELECT customer_tier, AVG(credit_limit) FROM customers GROUP BY customer_tier"),
    dict(id="fresh_and_age_by_country", q="and average age by country",
         gt="SELECT country_code, AVG(age) FROM customers GROUP BY country_code"),
    dict(id="fresh_what_about_urban", q="what about urban ratio",
         gt="SELECT AVG(urban_ratio) FROM geo_data"),

    # ---- real dangling follow-ups must still be refused -----------------
    dict(id="dangling_india", q="what about india", expect_needs_context=True),
    dict(id="dangling_by_device", q="also by device", expect_needs_context=True),
    dict(id="dangling_credit_card", q="what about credit card", expect_needs_context=True),
]


# --------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _num_eq(a: Any, b: Any) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()
    return abs(fa - fb) <= 1e-3 * max(1.0, abs(fa), abs(fb))


def rows_match(got: list[dict], want: list[tuple]) -> bool:
    """Same number of rows; each expected row's values all appear in a
    distinct returned row (extra columns in the answer are fine, column
    names and order are ignored)."""
    if len(got) != len(want):
        return False
    pool = [list(r.values()) for r in got]
    for w in want:
        for i, g in enumerate(pool):
            rest = list(g)
            ok = True
            for v in w:
                hit = next((j for j, x in enumerate(rest) if _num_eq(x, v)), None)
                if hit is None:
                    ok = False
                    break
                rest.pop(hit)
            if ok:
                pool.pop(i)
                break
        else:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="only run the ground-truth SQL")
    ap.add_argument("-k", default="", help="substring filter on id/question")
    args = ap.parse_args()

    cases = [c for c in CASES if args.k.lower() in (c["id"] + " " + c["q"]).lower()]
    conn = _connect()
    failures = 0

    if not args.dry:
        from ..pipeline import NEEDS_CONTEXT_REPLY, answer_question, warmup
        print("warming up the model ...")
        warmup()

    for c in cases:
        want_sets = []
        if "gt" in c:
            for sql in [c["gt"]] + c.get("alts", []):
                want_sets.append(conn.execute(sql).fetchall())
        if args.dry:
            first = want_sets[0] if want_sets else "(expects the 'ask it in full' reply)"
            print(f"[dry] {c['id']:28} {str(first)[:90]}")
            continue

        t = time.perf_counter()
        res = answer_question(c["q"])           # no session -> a brand-new chat
        secs = time.perf_counter() - t

        if c.get("expect_needs_context"):
            ok = res.text == NEEDS_CONTEXT_REPLY and res.sql is None
        else:
            ok = res.rows is not None and any(rows_match(res.rows, w) for w in want_sets)
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {c['id']:28} {secs:5.1f}s  {c['q']}")
        if not ok:
            print(f"        SQL     : {res.sql}")
            print(f"        answer  : {(res.text or '').splitlines()[0] if res.text else ''}")
            if want_sets:
                print(f"        expected: {str(want_sets[0])[:160]}")

    if not args.dry:
        print(f"\n{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
