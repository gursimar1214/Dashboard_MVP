"""
Tests for the synonym layer (synonyms.py). No Ollama, no network.

    python -m unittest chatbot.tests.test_synonyms -v      (from the project root)
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from .. import synonyms as sy
from ..value_fix import COLUMN_VALUES


def _audit_questions() -> list[str]:
    src = (Path(__file__).parent / "audit_questions.py").read_text(encoding="utf-8")
    return re.findall(r'\bq="([^"]+)"', src)


class TypoTests(unittest.TestCase):
    def test_fixes_near_misses_of_schema_words(self):
        text, fixed = sy.fix_typos("emal opning rate by tier")
        self.assertEqual(text, "email opening rate by tier")
        self.assertEqual(fixed, {"emal": "email", "opning": "opening"})
        self.assertEqual(sy.fix_typos("revnue by contry")[0], "revenue by country")

    def test_everyday_words_are_left_alone(self):
        for q in ("show as a table the revenue", "research the open rate", "what is the open rate",
                  "average spend per order", "gold customers"):
            self.assertEqual(sy.fix_typos(q)[0], q, q)

    def test_no_false_corrections_on_the_whole_audit_corpus(self):
        qs = _audit_questions()
        self.assertGreater(len(qs), 20)
        for q in qs:
            self.assertEqual(sy.fix_typos(q)[1], {}, f"typo-fixer changed: {q!r}")


class AnalyzeTests(unittest.TestCase):
    def test_contractions_do_not_block_slot_resolution(self):
        a = sy.analyze("What's our AOV in Germany this year?")
        self.assertEqual(a.metric, "aov")
        self.assertEqual(a.filters, {"country_code": "DE"})
        self.assertEqual(a.time, "this year")
        self.assertEqual(a.leftover, 0)

    def test_email_phrasings_all_mean_the_column(self):
        for q in ("which country opens the emails the most",
                  "what is the email open rate by country",
                  "average email engagement by tier",
                  "emal opening rate",
                  "email opens by region"):
            a = sy.analyze(q)
            self.assertEqual(a.metrics, ["email_open_rate"], q)
            self.assertTrue(any("customers.email_open_rate" in h for h in a.hints), q)

    def test_email_traffic_and_email_campaign_are_not_the_column(self):
        self.assertEqual(sy.analyze("sessions from email").filters, {"traffic_source": "email"})
        self.assertEqual(sy.analyze("email traffic by device").filters, {"traffic_source": "email"})
        a = sy.analyze("how many email campaigns are there")
        self.assertEqual((a.metrics, a.filters), (["campaigns_count"], {"campaign_type": "Email"}))
        a = sy.analyze("email campaign budget")
        self.assertEqual((a.metrics, a.filters, a.leftover), (["campaign_budget"], {"campaign_type": "Email"}, 0))

    def test_dimension_filters_time_rank(self):
        a = sy.analyze("revenue by payment method for gold customers in germany in 2023")
        self.assertEqual((a.metric, a.dimension, a.time), ("revenue", "payment_method", "2023"))
        self.assertEqual(a.filters, {"customer_tier": "Gold", "country_code": "DE"})
        self.assertEqual(a.leftover, 0)
        a = sy.analyze("top 5 countries by revenue")
        self.assertEqual((a.dimension, a.rank, a.top_n, a.leftover), ("country", "highest", 5, 0))
        a = sy.analyze("bounce rate by device in march 2023")
        self.assertEqual((a.dimension, a.time), ("device", "March 2023"))
        self.assertEqual(sy.analyze("and last month?").time, "last month")

    def test_two_letter_codes_only_in_capitals(self):
        self.assertEqual(sy.analyze("revenue in US").filters, {"country_code": "US"})
        self.assertEqual(sy.analyze("show us the revenue").filters, {})
        self.assertEqual(sy.analyze("revenue in india and us").filters, {"country_code": "IN"})  # "us" lowercase ignored
        self.assertEqual(sy.analyze("revenue in India and US").filters, {"country_code": "IN|US"})

    def test_average_order_value_is_not_also_orders(self):
        a = sy.analyze("average order value for gold customers in germany")
        self.assertEqual(a.metrics, ["aov"])
        self.assertFalse(a.complex)

    def test_multi_metric_and_comparisons_are_flagged_complex(self):
        self.assertTrue(sy.analyze("revenue and orders by country").complex)
        self.assertTrue(sy.analyze("customers who spent more than average").complex)
        self.assertTrue(sy.analyze("revenue in 2022 and 2023").complex)

    def test_unknown_words_are_counted_as_leftover(self):
        self.assertEqual(sy.analyze("revenue by country").leftover, 0)
        self.assertGreaterEqual(sy.analyze("revenue of unicorn customers").leftover, 1)
        self.assertGreaterEqual(sy.analyze("tell me a joke").leftover, 1)

    def test_region_hint_depends_on_the_metric_family(self):
        self.assertTrue(any("customers.region" in h for h in sy.analyze("average loyalty score by region").hints))
        self.assertTrue(any("sessions.geo_ip_region" in h for h in sy.analyze("bounce rate by region").hints))

    def test_clear_words(self):
        self.assertEqual(sy.analyze("overall").clear, {"*"})
        self.assertEqual(sy.analyze("what about all countries").clear, {"country_code"})


class VocabularySyncTests(unittest.TestCase):
    """If a valid value is added to the database/value_fix, the synonym layer
    must learn it too -- otherwise it would silently ignore that filter."""

    def test_every_valid_value_is_recognised(self):
        for tier in COLUMN_VALUES["customer_tier"]:
            self.assertEqual(sy.analyze(f"revenue for {tier.lower()} customers").filters, {"customer_tier": tier})
        for dev in COLUMN_VALUES["device_type"]:
            self.assertEqual(sy.analyze(f"sessions on {dev}").filters, {"device_type": dev})
        for pm in COLUMN_VALUES["payment_method"]:
            self.assertEqual(sy.analyze(f"revenue paid by {pm.replace('_', ' ')}").filters, {"payment_method": pm})
        for sp in COLUMN_VALUES["shipping_speed"]:
            self.assertEqual(sy.analyze(f"orders with {sp} shipping").filters, {"shipping_speed": sp})
        for ct in COLUMN_VALUES["campaign_type"]:
            self.assertEqual(sy.analyze(f"budget of {ct.lower()} campaigns").filters, {"campaign_type": ct})
        for ts in COLUMN_VALUES["traffic_source"]:
            self.assertEqual(sy.analyze(f"sessions from {ts}").filters, {"traffic_source": ts})
        for name, code in (("india", "IN"), ("germany", "DE"), ("france", "FR"), ("usa", "US"), ("britain", "UK")):
            self.assertEqual(sy.analyze(f"revenue in {name}").filters, {"country_code": code})

    def test_every_metric_has_a_label_and_a_group(self):
        for key, label, _ in sy._METRICS:
            self.assertTrue(label)
            self.assertIn(key, sy._METRIC_GROUP, key)


if __name__ == "__main__":
    unittest.main()
