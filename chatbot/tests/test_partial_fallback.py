"""Offline tests for the reviewed partial-answer fallback."""
from __future__ import annotations

import unittest
from unittest import mock

try:
    from .. import partial_fallback, pipeline
    from ..intent_schema import Intent
    _OK = True
except Exception:
    partial_fallback = pipeline = Intent = None
    _OK = False


@unittest.skipUnless(_OK, "pipeline dependencies not available")
class PartialFallbackTests(unittest.TestCase):
    def test_returns_an_explicit_subset_only_after_validation_and_review(self):
        plan = partial_fallback.PartialPlan(
            can_answer=True,
            answerable_part="Top 10 customers in India by purchase count",
            is_partial=True,
            omitted_parts=["the top 10 customers in the USA"],
            sql="SELECT customer_id, COUNT(*) AS purchases FROM transactions GROUP BY customer_id ORDER BY purchases DESC LIMIT 10",
        )
        with (
            mock.patch.object(partial_fallback, "propose", return_value=plan) as propose,
            mock.patch.object(partial_fallback, "review", return_value=partial_fallback.PartialReview(approved=True)) as review,
            mock.patch.object(pipeline, "validate_and_cap", side_effect=lambda sql: sql + " LIMIT 200") as validate,
            mock.patch.object(pipeline, "run_readonly_query", return_value=[{"customer_id": 17, "purchases": 8}]),
        ):
            answer = pipeline._try_partial_answer(
                "Show the top 10 customers in India and the USA by purchase count",
                "UNION query failed semantic review",
                "SELECT invalid",
                Intent.RANKING,
            )

        self.assertIsNotNone(answer)
        self.assertTrue(answer.is_partial)
        self.assertIn("Partial answer", answer.text)
        self.assertIn("couldn't safely answer: the top 10 customers in the USA", answer.text)
        self.assertEqual(answer.resolved_question, plan.answerable_part)
        self.assertEqual(propose.call_args.args[0], "Show the top 10 customers in India and the USA by purchase count")
        validate.assert_called_once_with(plan.sql)
        self.assertEqual(review.call_args.args[1].sql, plan.sql + " LIMIT 200")

    def test_rejects_a_plan_that_does_not_name_omitted_request_parts(self):
        plan = partial_fallback.PartialPlan(
            can_answer=True,
            answerable_part="Top 10 customers in India",
            is_partial=True,
            omitted_parts=[],
            sql="SELECT customer_id FROM customers LIMIT 10",
        )
        with mock.patch.object(partial_fallback, "propose", return_value=plan):
            answer = pipeline._try_partial_answer("top 10 in India and USA", "failed", "SELECT 1", Intent.RANKING)
        self.assertIsNone(answer)


if __name__ == "__main__":
    unittest.main()
