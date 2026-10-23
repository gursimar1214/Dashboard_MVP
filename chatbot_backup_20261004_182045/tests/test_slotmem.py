"""
Tests for structured slot memory (slotmem.py + memory.last_slots).
No Ollama, no network.

    python -m unittest chatbot.tests.test_slotmem -v      (from the project root)
"""
from __future__ import annotations

import unittest

from .. import memory, slotmem, synonyms


def _prev(question: str) -> slotmem.Slots:
    s = slotmem.slots_from_analysis(synonyms.analyze(question))
    assert s is not None, f"test setup: could not read {question!r}"
    return s


def _go(prev_q, q, has_history=True):
    return slotmem.resolve(q, _prev(prev_q) if prev_q else None, has_history)


class RenderTests(unittest.TestCase):
    def test_render_is_explicit_about_columns_and_values(self):
        s = _prev("revenue by payment method for gold customers in germany in 2022")
        self.assertEqual(
            s.render(),
            "revenue by payment method, where country_code = 'DE' and customer_tier = 'Gold', in 2022")

    def test_render_ranking_and_lists(self):
        self.assertEqual(_prev("top 5 countries by revenue").render(), "top 5 countries by revenue")
        self.assertEqual(_prev("which country opens the emails the most").render(),
                         "which country has the highest average email open rate")
        s = _prev("revenue in India and US")
        self.assertIn("country_code IN ('IN', 'US')", s.render())

    def test_dict_roundtrip(self):
        s = _prev("revenue by country for gold customers in 2022")
        self.assertEqual(slotmem.Slots.from_dict(s.to_dict()).render(), s.render())
        self.assertIsNone(slotmem.Slots.from_dict(None))
        self.assertIsNone(slotmem.Slots.from_dict({"metric": None}))


class DecisionTests(unittest.TestCase):
    def test_no_history_is_fresh_and_typo_fixed(self):
        r = _go(None, "emal opening rate by tier", has_history=False)
        self.assertEqual((r.mode, r.model_question), ("fresh", "email opening rate by tier"))
        self.assertEqual(r.resolved_text, "email opening rate by tier")     # shown: it was corrected

    def test_follow_up_edits_one_slot(self):
        r = _go("revenue in germany in 2022", "what about india")
        self.assertEqual(r.mode, "slots")
        self.assertEqual(r.model_question, "revenue, where country_code = 'IN', in 2022")
        self.assertFalse(r.use_history)                                     # history NOT sent

        r = _go("revenue in germany in 2022", "only gold customers")
        self.assertEqual(r.model_question, "revenue, where country_code = 'DE' and customer_tier = 'Gold', in 2022")

        r = _go("revenue in germany in 2022", "by payment method")
        self.assertEqual(r.model_question, "revenue by payment method, where country_code = 'DE', in 2022")

        r = _go("revenue in germany in 2022", "and last month?")
        self.assertEqual(r.model_question, "revenue, where country_code = 'DE', last month")

    def test_overall_clears_filters_and_grouping(self):
        self.assertEqual(_go("revenue by country for gold customers", "overall").model_question, "revenue")
        self.assertEqual(_go("revenue in germany for gold customers", "what about all countries").model_question,
                         "revenue, where customer_tier = 'Gold'")

    def test_new_metric_with_a_cue_keeps_filters(self):
        r = _go("revenue in germany for gold customers", "and orders?")
        self.assertEqual(r.model_question, "number of orders, where country_code = 'DE' and customer_tier = 'Gold'")

    def test_new_metric_without_a_cue_is_a_new_question(self):
        r = _go("revenue in germany for gold customers", "orders by country")
        self.assertEqual(r.mode, "fresh")
        self.assertEqual(r.model_question, "orders by country")             # old filters dropped
        self.assertFalse(r.use_history)

    def test_country_replaces_country_but_tier_is_added(self):
        r = _go("revenue in germany for gold customers", "what about france")
        self.assertEqual(r.model_question, "revenue, where country_code = 'FR' and customer_tier = 'Gold'")

    def test_email_follow_up_carries_the_hint(self):
        r = _go("average email open rate by country", "what about by tier")
        self.assertEqual(r.mode, "slots")
        self.assertEqual(r.model_question, "average email open rate by customer tier")
        self.assertTrue(any("customers.email_open_rate" in h for h in r.hints))

    def test_same_for_them_by_device_merges_customer_metric_slots(self):
        r = _go("average email open rate by customer tier for gold customers",
                "same for them by device")
        self.assertEqual(r.mode, "slots")
        self.assertEqual(r.slots.metric, "email_open_rate")
        self.assertEqual(r.slots.dimension, "device")
        self.assertEqual(r.slots.filters, {"customer_tier": "Gold"})

    def test_rank_follow_up(self):
        r = _go("sessions by device", "top 3")
        self.assertEqual(r.model_question, "top 3 device types by number of sessions")

    def test_uncertain_cases_fall_back_to_legacy(self):
        cases = [
            ("revenue in germany", "what about those above the average", "pronoun"),
            ("revenue in germany", "same for them by device", "pronoun"),
            ("revenue in germany", "tell me a joke", "not understood"),
            ("revenue in germany", "and compared to france", "complex"),
        ]
        for prev, q, why in cases:
            r = _go(prev, q)
            self.assertEqual(r.mode, "legacy", f"{q!r} ({why})")
            self.assertTrue(r.use_history)

    def test_follow_up_without_slots_from_last_turn_is_legacy(self):
        r = slotmem.resolve("what about india", None, has_history=True)
        self.assertEqual(r.mode, "legacy")

    def test_slots_stored_only_when_python_understood_everything(self):
        self.assertIsNotNone(_go(None, "revenue by country", False).slots)
        self.assertIsNone(_go(None, "revenue of unicorn customers", False).slots)
        self.assertIsNone(_go(None, "revenue and orders by country", False).slots)

    def test_chain_of_follow_ups(self):
        s = _prev("revenue in germany in 2022")
        for q, expected in [
            ("what about india", "revenue, where country_code = 'IN', in 2022"),
            ("only gold", "revenue, where country_code = 'IN' and customer_tier = 'Gold', in 2022"),
            ("by payment method", "revenue by payment method, where country_code = 'IN' and customer_tier = 'Gold', in 2022"),
        ]:
            r = slotmem.resolve(q, s, True)
            self.assertEqual(r.model_question, expected)
            s = r.slots


class MemoryIntegrationTests(unittest.TestCase):
    def test_last_slots_come_from_the_latest_turn_only(self):
        sid = memory.new_session_id()
        try:
            memory.remember(sid, "revenue in germany", "SELECT 1", "overview",
                            slots=_prev("revenue in germany").to_dict())
            self.assertEqual(memory.last_slots(sid)["metric"], "revenue")
            memory.remember(sid, "something odd", "SELECT 2", "overview", slots=None)
            self.assertIsNone(memory.last_slots(sid))      # newest turn unreadable -> no stale slots
        finally:
            memory.clear(sid)
        self.assertIsNone(memory.last_slots(sid))
        self.assertIsNone(memory.last_slots(None))


if __name__ == "__main__":
    unittest.main()
