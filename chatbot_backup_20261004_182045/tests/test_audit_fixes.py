"""
Regression tests for the audit fixes. No Ollama needed.

    python -m unittest chatbot.tests.test_audit_fixes -v      (from the project root)

Tests that need the database skip themselves if it is missing; tests that
need sqlglot (the validator) skip themselves if it is not installed. Install
the project requirements first so nothing is skipped:

    pip install -r chatbot/requirements.txt
"""
from __future__ import annotations

import os
import sqlite3
import unittest
from pathlib import Path

from .. import followup
from .. import schema_glossary as g
from ..config import DB_PATH, MEMORY_CONTEXT_CHARS

try:
    import sqlglot  # noqa: F401
    from ..sql_validator import SQLValidationError, validate_and_cap
    HAVE_SQLGLOT = True
except ImportError:
    HAVE_SQLGLOT = False

try:
    from .. import pipeline
    HAVE_PIPELINE = True
except ImportError:          # pydantic / ollama / httpx not installed
    HAVE_PIPELINE = False

HAVE_DB = Path(DB_PATH).exists()


def _db() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


# --------------------------------------------------------------------------
# Bug #1 -- memory budget
# --------------------------------------------------------------------------
class MemoryBudget(unittest.TestCase):
    @unittest.skipIf("CHATBOT_MEMORY_CONTEXT_CHARS" in os.environ, "overridden by env var")
    def test_default_budget_is_small(self):
        # was 3500 (up to ~1,100 prompt tokens re-read on every follow-up)
        self.assertLessEqual(MEMORY_CONTEXT_CHARS, 1500)

    def test_newest_turn_always_kept_and_budget_respected(self):
        from .. import memory
        T = memory._Turn
        sql = ("SELECT SUM(t.order_value) AS revenue FROM transactions t JOIN customers c "
               "ON c.customer_id = t.customer_id WHERE c.country_code = 'DE'")
        turns = [T(f"q{i}", f"meaning {i}", sql, "overview", "revenue=1.00") for i in range(6)]
        out = memory._render(turns)
        self.assertIn("Turn 6 (most recent)", out)
        self.assertLessEqual(len(out), MEMORY_CONTEXT_CHARS + 400)   # newest turn is exempt


# --------------------------------------------------------------------------
# Bug #2 -- follow-up guard
# --------------------------------------------------------------------------
class FollowupGuard(unittest.TestCase):
    STANDALONE = [
        "what about email open rate", "also credit limit by tier",
        "and average age by country", "what about urban ratio",
        "what about internet penetration", "and loyalty score by region",
        "also average review score", "what about campaign budget",
        "and session duration", "what about pages viewed", "and average income",
        "what about email engagement", "and the open rate", "what about age",
    ]
    # Real follow-ups: they name a filter value or a grouping, never a metric.
    DANGLING = [
        "what about india", "and last month?", "same for gold customers",
        "also by device", "also by tier", "and by country", "and by age",
        "also by  age", "what about credit card", "what about email",
        "what about overnight", "and for campaign type email", "what about paypal",
        "and mobile only", "what about ads", "also by shipping speed",
    ]

    def test_standalone_questions_are_not_refused_on_a_fresh_chat(self):
        for q in self.STANDALONE:
            with self.subTest(q=q):
                self.assertFalse(followup.looks_like_followup(q))
                self.assertFalse(followup.needs_earlier_turn(q, has_context=False))

    def test_dangling_followups_are_still_caught(self):
        for q in self.DANGLING:
            with self.subTest(q=q):
                self.assertTrue(followup.looks_like_followup(q))
                self.assertTrue(followup.needs_earlier_turn(q, has_context=False))

    def test_with_history_nothing_is_refused(self):
        for q in self.DANGLING + self.STANDALONE:
            self.assertFalse(followup.needs_earlier_turn(q, has_context=True))

    # Every column must be a conscious decision: a phrase that names it as a
    # metric, or None for ids / dates / filter values / groupings. Adding a
    # column to schema_glossary.ALLOWED_TABLES without adding it here fails
    # this test -- which is exactly how the original gap went unnoticed.
    COLUMN_PHRASES = {
        "customers.customer_id": None, "customers.age": "and average age by country",
        "customers.country_code": None, "customers.region": None,
        "customers.signup_date": None, "customers.loyalty_score": "what about loyalty score",
        "customers.email_open_rate": "what about email open rate",
        "customers.discount_usage_rate": "and discount usage rate",
        "customers.avg_review_score": "what about review score",
        "customers.referral_code": None, "customers.customer_tier": None,
        "customers.credit_limit": "also credit limit by tier",
        "sessions.session_id": None, "sessions.customer_id": None,
        "sessions.session_timestamp": None,
        "sessions.session_duration": "and session duration",
        "sessions.pages_viewed": "what about pages viewed",
        "sessions.cart_additions": "and cart additions",
        "sessions.bounce_flag": "and bounce rate", "sessions.traffic_source": None,
        "sessions.device_type": None, "sessions.campaign_id": None,
        "sessions.geo_ip_region": None,
        "transactions.transaction_id": None, "transactions.customer_id": None,
        "transactions.transaction_timestamp": None,
        "transactions.order_value": "and revenue",
        "transactions.items_count": "and items per order",
        "transactions.payment_method": None,
        "transactions.discount_applied": "and discount rate",
        "transactions.shipping_speed": None,
        "transactions.high_value_flag": "and high value orders",
        "geo_data.geo_ip_region": None, "geo_data.average_income": "and average income",
        "geo_data.urban_ratio": "what about urban ratio",
        "geo_data.internet_penetration": "what about internet penetration",
        "geo_data.region_tier": None,
        "marketing_campaigns.campaign_id": None, "marketing_campaigns.campaign_type": None,
        "marketing_campaigns.campaign_budget": "what about campaign budget",
        "marketing_campaigns.region_target": None,
        "marketing_campaigns.start_date": None, "marketing_campaigns.end_date": None,
    }

    def test_every_schema_column_has_a_decision(self):
        schema_cols = {f"{t}.{c}" for t, cols in g.ALLOWED_TABLES.items() for c in cols}
        self.assertEqual(schema_cols, set(self.COLUMN_PHRASES))

    def test_every_measurable_column_is_recognised_as_a_metric(self):
        for col, phrase in self.COLUMN_PHRASES.items():
            if phrase:
                with self.subTest(col=col):
                    self.assertFalse(followup.looks_like_followup(phrase), phrase)


# --------------------------------------------------------------------------
# Bugs #4 / #5 and the glossary gaps -- what the model reads
# --------------------------------------------------------------------------
class GlossaryText(unittest.TestCase):
    PROMPT_TEXT = "\n".join([g.METRICS_GLOSSARY, g.COMMON_MISTAKES, g.VALID_VALUES_TEXT])

    def test_campaign_join_is_no_longer_called_loosely_populated(self):
        self.assertNotIn("loosely", g.SCHEMA_DESCRIPTION)
        self.assertIn("sessions.campaign_id", g.SCHEMA_DESCRIPTION)

    def test_both_region_columns_are_explained(self):
        for needle in ("customers.region", "sessions.geo_ip_region", "HOME region"):
            self.assertIn(needle, g.COMMON_MISTAKES)
        self.assertIn("customers.region", g.SCHEMA_DESCRIPTION)   # relationships block

    def test_null_ranking_rule_present(self):
        self.assertIn("email_open_rate", g.COMMON_MISTAKES)
        self.assertIn("IS NOT NULL", g.COMMON_MISTAKES)

    def test_email_ambiguity_and_subquery_rules_present(self):
        self.assertIn("campaign_type = 'Email'", g.COMMON_MISTAKES)
        self.assertIn("ROW_NUMBER", g.COMMON_MISTAKES)

    def test_types_match_the_database(self):
        for line in ("age REAL", "session_duration REAL", "pages_viewed REAL"):
            self.assertIn(line, g.SCHEMA_DESCRIPTION)

    def test_no_undocumented_metric_columns(self):
        # every metric column the audit listed (plus the ones it missed) must
        # be mentioned by name somewhere the model actually reads
        must_mention = [
            "email_open_rate", "loyalty_score", "avg_review_score", "credit_limit",
            "age", "average_income", "urban_ratio", "internet_penetration",
            "campaign_budget", "campaign_type", "region_target",
            "session_duration", "pages_viewed", "cart_additions",
            "discount_usage_rate", "signup_date",
        ]
        for col in must_mention:
            with self.subTest(col=col):
                self.assertIn(col, self.PROMPT_TEXT)

    @unittest.skipUnless(HAVE_PIPELINE, "pipeline imports need pydantic/ollama")
    def test_prompt_still_fits_the_context_window(self):
        from ..config import LLM_NUM_CTX
        est_tokens = len(pipeline.FAST_SYSTEM_PROMPT) / 3.0       # pessimistic for SQL text
        reserve = 400 + 256 + 150                                  # history + output + question
        self.assertLess(est_tokens + reserve, LLM_NUM_CTX)

    @unittest.skipUnless(HAVE_PIPELINE, "pipeline imports need pydantic/ollama")
    def test_rate_like_columns_print_as_percentages(self):
        pv = pipeline._pretty_value
        self.assertEqual(pv("internet_penetration", 0.759), "75.9%")
        self.assertEqual(pv("email_open_rate", 0.3337), "33.4%")
        self.assertEqual(pv("urban_ratio", 0.616), "61.6%")


# --------------------------------------------------------------------------
# Facts in the glossary, checked against the real database
# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE_DB, f"database not found at {DB_PATH}")
class GlossaryMatchesDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = _db()

    @classmethod
    def tearDownClass(cls):
        cls.c.close()

    def one(self, sql):
        return self.c.execute(sql).fetchone()[0]

    def test_whitelist_matches_real_columns(self):
        for table, cols in g.ALLOWED_TABLES.items():
            real = {r[1] for r in self.c.execute(f"PRAGMA table_info({table})")}
            self.assertEqual(real, set(cols), table)

    def test_declared_real_types(self):
        types = {r[1]: r[2] for r in self.c.execute("PRAGMA table_info(customers)")}
        self.assertEqual(types["age"], "REAL")
        stypes = {r[1]: r[2] for r in self.c.execute("PRAGMA table_info(sessions)")}
        self.assertEqual(stypes["session_duration"], "REAL")
        self.assertEqual(stypes["pages_viewed"], "REAL")

    def test_every_documented_value_exists_and_nothing_else(self):
        from ..value_fix import COLUMN_VALUES
        owner = {"country_code": "customers", "customer_tier": "customers",
                 "traffic_source": "sessions", "device_type": "sessions",
                 "payment_method": "transactions", "shipping_speed": "transactions",
                 "high_value_flag": "transactions", "campaign_type": "marketing_campaigns",
                 "region_tier": "geo_data"}
        for col, values in COLUMN_VALUES.items():
            real = {r[0] for r in self.c.execute(f"SELECT DISTINCT {col} FROM {owner[col]}")}
            self.assertEqual(real, set(values), col)

    def test_campaign_join_is_fully_populated(self):          # bug #4
        self.assertEqual(self.one("SELECT COUNT(*) FROM sessions WHERE campaign_id IS NULL"), 0)
        self.assertEqual(self.one(
            "SELECT COUNT(*) FROM sessions s LEFT JOIN marketing_campaigns m "
            "ON m.campaign_id = s.campaign_id WHERE m.campaign_id IS NULL"), 0)

    def test_customer_region_joins_geo_data(self):
        self.assertEqual(self.one(
            "SELECT COUNT(*) FROM customers WHERE region NOT IN (SELECT geo_ip_region FROM geo_data)"), 0)

    def test_two_region_columns_really_disagree(self):         # bug #5
        diff, total = self.c.execute(
            "SELECT SUM(c.region <> s.geo_ip_region), COUNT(*) FROM sessions s "
            "JOIN customers c ON c.customer_id = s.customer_id").fetchone()
        self.assertGreater(diff / total, 0.95)

    def test_null_shares_quoted_in_the_prompt(self):
        n = self.one("SELECT COUNT(*) FROM customers")
        self.assertAlmostEqual(self.one("SELECT SUM(email_open_rate IS NULL) FROM customers") / n, 0.11, delta=0.02)
        self.assertAlmostEqual(self.one("SELECT SUM(age IS NULL) FROM customers") / n, 0.05, delta=0.02)
        m = self.one("SELECT COUNT(*) FROM sessions")
        self.assertAlmostEqual(self.one("SELECT SUM(pages_viewed IS NULL) FROM sessions") / m, 0.05, delta=0.02)

    def test_ascending_sort_puts_nulls_first(self):            # why rule 10 exists
        first = self.one("SELECT email_open_rate FROM customers ORDER BY email_open_rate ASC LIMIT 1")
        self.assertIsNone(first)
        first = self.one("SELECT email_open_rate FROM customers WHERE email_open_rate IS NOT NULL "
                         "ORDER BY email_open_rate ASC LIMIT 1")
        self.assertIsNotNone(first)

    def test_customer_discount_column_differs_from_order_level_rate(self):
        a = self.one("SELECT AVG(discount_usage_rate) FROM customers")
        b = self.one("SELECT AVG(CASE WHEN discount_applied THEN 1.0 ELSE 0.0 END) FROM transactions")
        self.assertGreater(abs(a - b), 0.1)

    def test_every_few_shot_sql_runs_on_the_database(self):
        for question, sql in pipeline.FEW_SHOT_EXAMPLES if HAVE_PIPELINE else []:
            with self.subTest(q=question):
                rows = self.c.execute(sql).fetchall()
                self.assertGreaterEqual(len(rows), 1)

    @unittest.skipUnless(HAVE_PIPELINE, "needs pipeline")
    def test_followup_worked_examples_run(self):
        sqls = [pipeline.FOLLOWUP_BASE_SQL] + [s for _, _, s in pipeline.FOLLOWUP_EXAMPLES_LIST]
        for sql in sqls:
            self.c.execute(sql).fetchall()


# --------------------------------------------------------------------------
# Bug #3 -- validator and derived tables (needs sqlglot)
# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SQLGLOT, "sqlglot not installed -- run: pip install -r chatbot/requirements.txt")
class ValidatorDerivedTables(unittest.TestCase):
    ACCEPT = {
        "derived table, qualified alias":
            "SELECT s.customer_id, s.total FROM (SELECT customer_id, SUM(order_value) AS total "
            "FROM transactions GROUP BY customer_id) AS s ORDER BY s.total DESC LIMIT 5",
        "derived table, outer WHERE uses inner alias (no qualifier)":
            "SELECT customer_id FROM (SELECT customer_id, SUM(order_value) AS total "
            "FROM transactions GROUP BY customer_id) AS s WHERE total > 1000",
        "derived table without AS":
            "SELECT COUNT(*) AS n FROM (SELECT customer_id, SUM(order_value) AS total "
            "FROM transactions GROUP BY customer_id) s WHERE s.total > 1000",
        "above-average via derived tables + scalar subquery":
            "SELECT COUNT(*) AS customers FROM (SELECT customer_id, SUM(order_value) AS s FROM transactions "
            "GROUP BY customer_id) WHERE s > (SELECT AVG(s2) FROM (SELECT SUM(order_value) AS s2 "
            "FROM transactions GROUP BY customer_id))",
        "derived table joined to a real table":
            "SELECT c.country_code, x.total FROM customers c JOIN (SELECT customer_id, SUM(order_value) "
            "AS total FROM transactions GROUP BY customer_id) x ON x.customer_id = c.customer_id LIMIT 10",
        "window function top-N per group":
            "SELECT customer_tier, customer_id, revenue FROM (SELECT c.customer_tier, c.customer_id, "
            "SUM(t.order_value) AS revenue, ROW_NUMBER() OVER (PARTITION BY c.customer_tier ORDER BY "
            "SUM(t.order_value) DESC) AS rn FROM customers c JOIN transactions t ON t.customer_id = "
            "c.customer_id GROUP BY c.customer_tier, c.customer_id) AS r WHERE rn <= 3",
        "CTE":
            "WITH m AS (SELECT substr(transaction_timestamp,1,7) AS month, SUM(order_value) AS revenue "
            "FROM transactions GROUP BY month) SELECT month, revenue FROM m ORDER BY month",
        "same alias reused in two scopes":
            "SELECT c.country_code FROM customers c WHERE c.customer_id IN "
            "(SELECT c.customer_id FROM transactions c WHERE c.order_value > 100) LIMIT 5",
        "NOT IN subquery":
            "SELECT COUNT(*) AS customers FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM sessions)",
    }
    REJECT = {
        "unknown table":                "SELECT * FROM users",
        "unknown table hidden in a derived table":
            "SELECT * FROM (SELECT * FROM users) AS u",
        "unknown column on a real table": "SELECT c.nonsense FROM customers c",
        "unknown unqualified column":     "SELECT nonsense FROM customers",
        "unknown alias":                  "SELECT x.customer_id FROM customers c",
        "metadata table":                 "SELECT name FROM sqlite_master",
        "write statement":                "DELETE FROM customers",
        "two statements":                 "SELECT 1; SELECT 2",
        "placeholder":                    "SELECT * FROM customers WHERE customer_id = ?",
    }

    def test_accepted(self):
        for name, sql in self.ACCEPT.items():
            with self.subTest(name):
                validate_and_cap(sql)               # must not raise

    def test_rejected(self):
        for name, sql in self.REJECT.items():
            with self.subTest(name):
                with self.assertRaises(SQLValidationError):
                    validate_and_cap(sql)

    def test_row_cap(self):
        from ..config import MAX_ROWS_RETURNED
        self.assertIn(f"LIMIT {MAX_ROWS_RETURNED}", validate_and_cap("SELECT * FROM customers").upper())
        self.assertIn(f"LIMIT {MAX_ROWS_RETURNED}", validate_and_cap("SELECT * FROM customers LIMIT 99999").upper())
        self.assertIn("LIMIT 7", validate_and_cap("SELECT * FROM customers LIMIT 7").upper())

    @unittest.skipUnless(HAVE_PIPELINE, "needs pipeline")
    def test_all_few_shot_sql_passes_validation(self):
        for question, sql in pipeline.FEW_SHOT_EXAMPLES:
            with self.subTest(q=question):
                validate_and_cap(sql)

    @unittest.skipUnless(HAVE_DB, "database not found")
    def test_derived_table_query_runs_end_to_end(self):
        sql = validate_and_cap(self.ACCEPT["above-average via derived tables + scalar subquery"])
        c = _db()
        try:
            n = c.execute(sql).fetchone()[0]
        finally:
            c.close()
        self.assertGreater(n, 0)


# --------------------------------------------------------------------------
# The live-question runner's matcher
# --------------------------------------------------------------------------
class RowMatcher(unittest.TestCase):
    def test_matching(self):
        from .audit_questions import rows_match
        self.assertTrue(rows_match([{"x": 0.33372}], [(0.3337228924,)]))
        self.assertTrue(rows_match([{"r": "region_57", "v": 0.3446, "n": 3000}], [("region_57", 0.34462520)]))
        self.assertTrue(rows_match([{"a": 2, "b": "Gold"}, {"a": 1, "b": "Bronze"}], [("Bronze", 1), ("Gold", 2)]))
        self.assertFalse(rows_match([{"r": "region_1", "v": 0.3446}], [("region_57", 0.3446)]))
        self.assertFalse(rows_match([{"x": 1}, {"x": 2}], [(1,)]))        # wrong row count
        self.assertFalse(rows_match([{"x": None}], [(0.33,)]))            # NULL row first


if __name__ == "__main__":
    unittest.main()
