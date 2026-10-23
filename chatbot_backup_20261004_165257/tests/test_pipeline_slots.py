"""
Pipeline-level test of the synonym layer + slot memory + answer ids.

The model is replaced by a stub (so no Ollama) and the SQL validator by a
pass-through (it has its own tests); the real database, memory, caches and
reply formatting run. Skipped when the dependencies or the database are
missing.

    python -m unittest chatbot.tests.test_pipeline_slots -v
"""
from __future__ import annotations

import os
import unittest
import uuid
from unittest import mock

try:
    from .. import memory, pipeline
    from ..config import DB_PATH
    _OK = os.path.exists(DB_PATH)
except Exception:        # pydantic / sqlglot / ollama not installed here
    pipeline = None
    _OK = False

SQL = ("SELECT customer_tier, AVG(email_open_rate) AS email_open_rate "
       "FROM customers GROUP BY customer_tier ORDER BY email_open_rate DESC")


class _Plan:
    def __init__(self, sql, q=None):
        self.sql, self.standalone_question = sql, q


@unittest.skipUnless(_OK, "pipeline dependencies or database not available")
class PipelineSlotTests(unittest.TestCase):
    def setUp(self):
        self.seen: list[tuple[str, str]] = []
        outer = self

        def fake_generate(system, prompt, schema, max_tokens=300, name="x"):
            outer.seen.append((schema.__name__, prompt))
            return _Plan(SQL, "resolved by the model")

        self.patches = [
            mock.patch.object(pipeline.llm, "generate_structured", fake_generate),
            mock.patch.object(pipeline, "validate_and_cap", lambda s: s + " LIMIT 200"),
        ]
        for p in self.patches:
            p.start()
        self.sid = uuid.uuid4().hex

    def tearDown(self):
        for p in self.patches:
            p.stop()
        memory.clear(self.sid)

    def ask(self, q, sid=None):
        self.seen.clear()
        return pipeline.answer_question(q, session_id=sid or self.sid)

    def test_first_question_gets_the_email_hint_and_no_history(self):
        r = self.ask("which country opens the emails the most")
        self.assertEqual(self.seen, [])  # known customer metric/dimension is a SQL template
        self.assertIn("AVG(c.email_open_rate)", r.sql)
        self.assertTrue(r.answer_id)

    def test_follow_up_is_merged_in_python_without_history(self):
        self.ask("average email open rate by country for gold customers in %s" % ("2023" if self.sid[0] < "8" else "2024"))
        r = self.ask("what about france")
        schema, prompt = self.seen[0]
        self.assertEqual(schema, "FastPlanNoCtx")                    # model only writes SQL
        self.assertNotIn("Conversation so far", prompt)
        self.assertIn("country_code = 'FR'", prompt)
        self.assertIn("customer_tier = 'Gold'", prompt)               # earlier filter kept
        self.assertIn("country_code = 'FR'", r.interpreted_as)         # shown to the user

    def test_same_for_them_by_dimension_is_resolved_without_history(self):
        self.ask("average loyalty score by customer tier")
        self.ask("same for them by device")
        schema, prompt = self.seen[0]
        self.assertEqual(schema, "FastPlanNoCtx")
        self.assertNotIn("Conversation so far", prompt)

    def test_complete_above_average_question_uses_template_even_with_history(self):
        self.ask("average loyalty score by customer tier")
        self.seen.clear()
        r = self.ask("customers whose total spend is above the overall average")
        self.assertEqual(self.seen, [])
        self.assertIn("AVG(customer_spend)", r.sql)
        self.assertNotIn("AVG(order_value)", r.sql)

    def test_dangling_follow_up_still_asks_for_the_full_question(self):
        r = self.ask("and last month?", sid=uuid.uuid4().hex)
        self.assertEqual(self.seen, [])
        self.assertEqual(r.intent, "followup")

    def test_every_answer_gets_a_distinct_answer_id(self):
        a = self.ask("revenue by payment method for gold customers (t5 %s)" % self.sid[:6])
        b = self.ask("revenue by payment method for gold customers (t5 %s)" % self.sid[:6])   # cache hit
        self.assertTrue(a.answer_id and b.answer_id)
        self.assertNotEqual(a.answer_id, b.answer_id)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(pipeline is not None, "pipeline dependencies not available")
class IntentGuessTests(unittest.TestCase):
    def test_join_condition_is_not_a_customer_lookup(self):
        sql = ("SELECT AVG(t.order_value) FROM transactions AS t JOIN customers AS c "
               "ON c.customer_id = t.customer_id WHERE c.customer_tier = 'Gold'")
        self.assertNotEqual(pipeline._guess_intent(sql).value, "lookup")

    def test_a_named_customer_is_a_lookup(self):
        self.assertEqual(pipeline._guess_intent("SELECT * FROM customers WHERE customer_id = 'C42'").value, "lookup")
        self.assertEqual(pipeline._guess_intent("SELECT * FROM customers WHERE customer_id = 42").value, "lookup")


class ConfigTests(unittest.TestCase):
    def test_localhost_is_swapped_for_the_ip(self):
        from .. import config
        self.assertNotIn("localhost", config.OLLAMA_HOST)
        self.assertGreaterEqual(config.HTTP_KEEPALIVE_S, 60)
