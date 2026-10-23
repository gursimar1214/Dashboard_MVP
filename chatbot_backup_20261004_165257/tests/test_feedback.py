"""
Tests for the feedback loop (feedback.py, feedback_review.py). No network.

    python -m unittest chatbot.tests.test_feedback -v      (from the project root)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .. import feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "fb.jsonl"
        self.patch = mock.patch.object(feedback, "FEEDBACK_LOG", self.log)
        self.patch.start()
        feedback._recent.clear()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_thumbs_down_is_logged_with_the_question_and_sql(self):
        feedback.remember_answer("aa11", trace_id=None, question="which region has the most emails",
                                 sql="SELECT 1", path="llm", resolution="fresh", model="m")
        out = feedback.save_feedback("aa11", False, "wrong column")
        self.assertTrue(out["logged"])
        self.assertTrue(out["known_answer"])
        self.assertFalse(out["langfuse"])                       # tracing off: still logged locally
        line = json.loads(self.log.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((line["helpful"], line["comment"], line["sql"]), (False, "wrong column", "SELECT 1"))
        self.assertEqual(line["question"], "which region has the most emails")

    def test_changed_mind_wins_in_the_review(self):
        feedback.remember_answer("bb22", question="q")
        feedback.save_feedback("bb22", False)
        feedback.save_feedback("bb22", True)
        final = feedback.latest_per_answer(feedback.read_log())
        self.assertEqual(len(final), 1)
        self.assertTrue(final[0]["helpful"])

    def test_langfuse_score_sent_when_the_answer_has_a_trace(self):
        feedback.remember_answer("cc33", trace_id="cc33", question="q")
        with mock.patch.object(feedback.observability, "score", return_value=True) as sc:
            out = feedback.save_feedback("cc33", True, "great")
        sc.assert_called_once_with("cc33", "user_feedback", 1.0, comment="great")
        self.assertTrue(out["langfuse"])

    def test_unknown_answer_still_logs(self):
        out = feedback.save_feedback("dd44", False)
        self.assertTrue(out["logged"])
        self.assertFalse(out["known_answer"])

    def test_only_the_most_recent_answers_are_remembered(self):
        with mock.patch.object(feedback, "_KEEP", 3):
            for i in range(5):
                feedback.remember_answer(f"id{i}", question=str(i))
        self.assertEqual(list(feedback._recent), ["id2", "id3", "id4"])

    def test_write_error_does_not_raise(self):
        with mock.patch.object(feedback, "FEEDBACK_LOG", Path(self.tmp.name) / "no" / "such" / "\0bad"):
            out = feedback.save_feedback("ee55", False)
        self.assertFalse(out["logged"])


if __name__ == "__main__":
    unittest.main()
