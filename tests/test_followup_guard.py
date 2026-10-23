"""
Tests for chatbot/followup.py -- stdlib only.

    python -m unittest tests.test_followup_guard -v

Two-sided on purpose: it must catch dangling follow-ups, and it must NEVER
block a real question (a false "please ask in full" is as bad as a guess).
"""
import unittest

from chatbot import followup


DANGLING = [
    "and last month?",
    "And last month",
    "what about india",
    "What about India?",
    "how about germany",
    "same for gold customers",
    "same but for 2023",
    "the same for france",
    "also by device",
    "and only mobile?",
    "and for silver customers",
    "those in france",
    "instead of germany use india",
    "what if we only look at 2022",
]

REAL_QUESTIONS = [
    "how many gold customers are in france",
    "What's our AOV in Germany this year?",
    "Which payment method brings in the most revenue?",
    "revenue by country",
    "top 5 customers by number of purchases",
    "list customers in india",
    "show me orders by shipping speed",
    "and how many customers are in france",            # cue + full-question opener
    "also show orders by country",                     # cue + metric
    "what about revenue in france",                    # cue + metric
    "and what is the bounce rate on mobile",
    "monthly orders in 2022",
    "compare gold and platinum",
    "customers",
    "",
    # long message that happens to start with a cue: a real question
    "and then please tell me something about all of the customers that joined last year in total",
]


class Detection(unittest.TestCase):
    def test_dangling_followups_are_caught(self):
        for q in DANGLING:
            with self.subTest(q=q):
                self.assertTrue(followup.looks_like_followup(q), q)

    def test_real_questions_are_never_blocked(self):
        for q in REAL_QUESTIONS:
            with self.subTest(q=q):
                self.assertFalse(followup.looks_like_followup(q), q)

    def test_none_is_safe(self):
        self.assertFalse(followup.looks_like_followup(None))  # type: ignore[arg-type]


class NeedsEarlierTurn(unittest.TestCase):
    def test_history_present_never_blocks(self):
        for q in DANGLING + REAL_QUESTIONS:
            self.assertFalse(followup.needs_earlier_turn(q, has_context=True), q)
            self.assertFalse(followup.needs_earlier_turn(q, has_context=True, model_says_followup=True), q)

    def test_no_history_dangling_blocks_before_the_model(self):
        for q in DANGLING:
            with self.subTest(q=q):
                self.assertTrue(followup.needs_earlier_turn(q, has_context=False), q)

    def test_no_history_real_question_passes(self):
        for q in REAL_QUESTIONS:
            with self.subTest(q=q):
                self.assertFalse(followup.needs_earlier_turn(q, has_context=False), q)

    def test_model_label_is_trusted_only_for_short_messages(self):
        self.assertTrue(followup.needs_earlier_turn("only mobile", False, model_says_followup=True))
        self.assertFalse(followup.needs_earlier_turn("only mobile", False, model_says_followup=False))
        long_q = "total revenue from gold customers in france during 2022 by month"
        self.assertFalse(followup.needs_earlier_turn(long_q, False, model_says_followup=True))


if __name__ == "__main__":
    unittest.main()
