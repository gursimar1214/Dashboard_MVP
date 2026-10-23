"""
End-to-end tests of the follow-up flow through pipeline.answer_question().

The language model is replaced by a scripted stand-in, so these tests check
everything AROUND the model deterministically: what it is shown each turn,
what gets remembered, what is cached, and what happens when there is nothing
to follow up on. The SQL validator is replaced by a simple stand-in too (it
has its own tests); the database is a real SQLite file built from the schema.
They do NOT measure how well Qwen follows the instructions -- use
tests/run_followup_golden.py against a live Ollama for that.

    python -m unittest tests.test_followup_flow -v
"""
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from chatbot import db, memory, pipeline
from chatbot.config import LLM_NUM_CTX, SQL_MAX_TOKENS
from chatbot.intent_schema import Intent
from chatbot.schema_glossary import ALLOWED_TABLES
from chatbot.sql_validator import SQLValidationError

_tmpdir = None
_db_path = None


# --------------------------------------------------------------------------
# Test database: real tables from the schema whitelist, a few rows each
# --------------------------------------------------------------------------

def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    for table, cols in ALLOWED_TABLES.items():
        conn.execute(f"CREATE TABLE {table} ({', '.join(sorted(cols))})")
    customers = [
        (1, "DE", "Gold"), (2, "DE", "Silver"), (3, "IN", "Gold"),
        (4, "IN", "Bronze"), (5, "FR", "Platinum"),
    ]
    for cid, cc, tier in customers:
        conn.execute("INSERT INTO customers (customer_id, country_code, customer_tier) VALUES (?,?,?)", (cid, cc, tier))
    tx = [  # customer, timestamp, value, payment
        (1, "2022-03-01 10:00:00", 100.0, "paypal"), (2, "2022-06-01 10:00:00", 50.0, "credit_card"),
        (3, "2022-07-01 10:00:00", 70.0, "paypal"), (4, "2022-08-01 10:00:00", 30.0, "crypto"),
        (1, "2023-02-01 10:00:00", 200.0, "debit_card"), (5, "2023-05-01 10:00:00", 10.0, "paypal"),
    ]
    for i, (cid, ts, val, pay) in enumerate(tx, 1):
        conn.execute(
            "INSERT INTO transactions (transaction_id, customer_id, transaction_timestamp, order_value, payment_method) "
            "VALUES (?,?,?,?,?)", (i, cid, ts, val, pay))
    conn.execute("INSERT INTO sessions (session_id, customer_id, traffic_source) VALUES (1, 1, 'ads')")
    conn.execute("INSERT INTO sessions (session_id, customer_id, traffic_source) VALUES (2, 3, 'organic')")
    conn.commit()
    conn.close()


def setUpModule():
    global _tmpdir, _db_path
    _tmpdir = tempfile.TemporaryDirectory()
    _db_path = Path(_tmpdir.name) / "test.db"
    _build_db(_db_path)


def tearDownModule():
    _tmpdir.cleanup()


# --------------------------------------------------------------------------
# Stand-ins
# --------------------------------------------------------------------------

def fake_validate(sql: str) -> str:
    """Stands in for sql_validator.validate_and_cap: rejects non-SELECT and
    injects the row cap, like the real one."""
    sql = sql.strip().rstrip(";")
    if not sql.upper().startswith("SELECT"):
        raise SQLValidationError("only SELECT statements are allowed")
    if " LIMIT " not in sql.upper():
        sql += " LIMIT 200"
    return sql


def plan(intent, standalone, sql):
    return pipeline.FastPlan(intent=Intent(intent), standalone_question=standalone, sql=sql)


class ScriptedLLM:
    """Replaces llm.generate_structured. Returns the next scripted plan and
    records every prompt it was shown."""

    def __init__(self, *plans, delay: float = 0.0):
        self.plans = list(plans)
        self.prompts: list[str] = []
        self.delay = delay
        self._lock = threading.Lock()

    def __call__(self, system, prompt, schema, max_tokens=300):
        with self._lock:
            self.prompts.append(prompt)
            nxt = self.plans.pop(0)
        if self.delay:
            time.sleep(self.delay)
        return nxt


SQL_DE_2022 = pipeline.FOLLOWUP_BASE_SQL
SQL_IN_2022 = pipeline.FOLLOWUP_EXAMPLES_LIST[0][2]
SQL_IN_2022_GOLD = (
    "SELECT SUM(t.order_value) AS revenue FROM transactions t JOIN customers c ON c.customer_id = t.customer_id "
    "WHERE c.country_code = 'IN' AND c.customer_tier = 'Gold' AND t.transaction_timestamp >= '2022-01-01' "
    "AND t.transaction_timestamp < '2023-01-01'"
)


class FlowTestCase(unittest.TestCase):
    def setUp(self):
        with memory._lock:
            memory._sessions.clear()
            memory._gates.clear()
        with pipeline._cache_lock:
            pipeline._cache.clear()
        with db._ref_lock:
            db._ref_cache.update(value=None, expires_at=0.0)
        patches = [
            mock.patch.object(db, "DB_PATH", _db_path),
            mock.patch.object(pipeline, "validate_and_cap", side_effect=fake_validate),
            mock.patch.object(pipeline, "FAST_MODE", True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def ask(self, llm_script, question, session="s1"):
        with mock.patch.object(pipeline.llm, "generate_structured", llm_script):
            return pipeline.answer_question(question, session_id=session)


# --------------------------------------------------------------------------
# The prompt material itself
# --------------------------------------------------------------------------

class PromptMaterial(FlowTestCase):
    def test_every_worked_example_sql_runs_on_the_schema(self):
        """The follow-up examples shown to the model must be valid SQL on the
        real columns -- a wrong example teaches the model a wrong pattern."""
        sqls = [pipeline.FOLLOWUP_BASE_SQL] + [s for _, _, s in pipeline.FOLLOWUP_EXAMPLES_LIST]
        sqls += [sql for _, sql in pipeline.FEW_SHOT_EXAMPLES]
        for sql in sqls:
            with self.subTest(sql=sql[:70]):
                db.run_readonly_query(sql)  # raises on unknown column / syntax error

    def test_worked_examples_give_the_answers_they_claim(self):
        rev = lambda sql: db.run_readonly_query(sql)[0]["revenue"]
        self.assertEqual(rev(SQL_DE_2022), 150.0)           # DE, 2022
        self.assertEqual(rev(pipeline.FOLLOWUP_EXAMPLES_LIST[0][2]), 100.0)   # IN, 2022
        self.assertEqual(rev(pipeline.FOLLOWUP_EXAMPLES_LIST[1][2]), 100.0)   # DE gold, 2022
        self.assertEqual(rev(pipeline.FOLLOWUP_EXAMPLES_LIST[2][2]), 200.0)   # DE, 2023
        self.assertEqual(rev(pipeline.FOLLOWUP_EXAMPLES_LIST[4][2]), 250.0)   # all countries, 2022

    def test_worst_case_prompt_fits_the_context_window(self):
        """Longest realistic prompt: a max-length question, a full history,
        the follow-up rules and examples, plus the model's answer. Uses a
        pessimistic 3 characters per token."""
        big_sql = "SELECT " + ", ".join(f"c{i}" for i in range(180))
        for i in range(memory.MAX_TURNS_PER_SESSION):
            memory.remember("w", f"question {i} " + "x" * 250, big_sql, "overview", summary="v" * 400)
        ctx = memory.get_context("w")
        total_chars = (
            len(pipeline.FAST_SYSTEM_PROMPT) + len(pipeline.CONTEXT_INSTRUCTION)
            + len(pipeline.FOLLOWUP_EXAMPLES) + len(ctx) + 2000 + 200
        )
        self.assertLess(total_chars / 3 + SQL_MAX_TOKENS, LLM_NUM_CTX,
                        f"{total_chars} chars would not fit in num_ctx={LLM_NUM_CTX}")

    def test_first_question_prompt_carries_no_followup_material(self):
        llm = ScriptedLLM(plan("overview", "total revenue in germany in 2022", SQL_DE_2022))
        self.ask(llm, "total revenue in germany in 2022")
        self.assertNotIn("Worked examples", llm.prompts[0])
        self.assertNotIn("Conversation so far", llm.prompts[0])
        self.assertIn("Question: total revenue in germany in 2022", llm.prompts[0])


# --------------------------------------------------------------------------
# Follow-up chains
# --------------------------------------------------------------------------

class Chains(FlowTestCase):
    def _three_turns(self):
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("overview", "total revenue in india in 2022", SQL_IN_2022),
            plan("overview", "total revenue in india in 2022 from gold customers", SQL_IN_2022_GOLD),
        )
        r1 = self.ask(llm, "total revenue in germany in 2022")
        r2 = self.ask(llm, "what about india")
        r3 = self.ask(llm, "and only gold customers?")
        return llm, r1, r2, r3

    def test_chain_gives_right_rows_each_turn(self):
        _, r1, r2, r3 = self._three_turns()
        self.assertEqual(r1.rows, [{"revenue": 150.0}])
        self.assertEqual(r2.rows, [{"revenue": 100.0}])
        self.assertEqual(r3.rows, [{"revenue": 70.0}])

    def test_second_turn_is_shown_the_first(self):
        llm, *_ = self._three_turns()
        p2 = llm.prompts[1]
        self.assertIn(pipeline.CONTEXT_INSTRUCTION, p2)
        self.assertIn("Worked examples", p2)
        self.assertIn("Conversation so far:", p2)
        self.assertIn("User typed: total revenue in germany in 2022", p2)
        self.assertIn("SQL that ran: " + SQL_DE_2022, p2)
        self.assertIn("Result: revenue=150.00", p2)
        self.assertIn("NEW question: what about india", p2)
        self.assertIn("Reference date", p2)

    def test_third_turn_sees_the_resolved_meaning_of_the_second(self):
        """The point of the rewrite: turn 3 must not have to decode the
        bare words 'what about india'."""
        llm, *_ = self._three_turns()
        p3 = llm.prompts[2]
        self.assertIn("User typed: what about india", p3)
        self.assertIn("Meaning (self-contained): total revenue in india in 2022", p3)
        self.assertIn("SQL that ran: " + SQL_IN_2022, p3)
        self.assertIn("MOST RECENT", p3)
        history = p3.split("Conversation so far:")[1]          # rules/examples above mention these words too
        self.assertLess(history.index("total revenue in germany in 2022"), history.index("what about india"))

    def test_memory_stores_sql_as_written_not_with_the_injected_row_cap(self):
        llm, r1, *_ = self._three_turns()
        self.assertTrue(r1.sql.endswith("LIMIT 200"))            # what ran / what the UI shows
        self.assertNotIn("LIMIT 200", llm.prompts[1])            # what the model is shown as history
        self.assertNotIn("LIMIT 200", llm.prompts[2])

    def test_interpreted_as(self):
        _, r1, r2, r3 = self._three_turns()
        self.assertIsNone(r1.interpreted_as)                     # no history, taken as typed
        self.assertEqual(r2.interpreted_as, "total revenue in india in 2022")
        self.assertEqual(r3.interpreted_as, "total revenue in india in 2022 from gold customers")
        self.assertIn("interpreted_as", r2.to_dict())
        self.assertEqual(r2.to_dict()["answer"], r2.text)        # API shape otherwise unchanged
        self.assertEqual(set(r2.to_dict()), {"answer", "sql", "rows", "intent", "interpreted_as"})

    def test_standalone_question_equal_to_typed_is_not_reported(self):
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("overview", "how many sessions came from ads",
                 "SELECT COUNT(*) AS sessions FROM sessions WHERE traffic_source = 'ads'"),
        )
        self.ask(llm, "total revenue in germany in 2022")
        r = self.ask(llm, "How many sessions came from ads")
        self.assertIsNone(r.interpreted_as)

    def test_unrelated_question_after_history_still_answers(self):
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("overview", "how many sessions came from ads",
                 "SELECT COUNT(*) AS sessions FROM sessions WHERE traffic_source = 'ads'"),
        )
        self.ask(llm, "total revenue in germany in 2022")
        r = self.ask(llm, "how many sessions came from ads")
        self.assertEqual(r.rows, [{"sessions": 1}])

    def test_intent_followup_with_history_and_sql_is_answered(self):
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("followup", "total revenue in india in 2022", SQL_IN_2022),
        )
        self.ask(llm, "total revenue in germany in 2022")
        r = self.ask(llm, "what about india")
        self.assertEqual(r.rows, [{"revenue": 100.0}])


# --------------------------------------------------------------------------
# Nothing to follow up on
# --------------------------------------------------------------------------

class NothingToFollow(FlowTestCase):
    def test_dangling_first_message_asks_and_never_calls_the_model(self):
        llm = ScriptedLLM()                       # any call would pop from an empty list
        for q in ["and last month?", "what about india", "same for gold customers"]:
            r = self.ask(llm, q)
            self.assertEqual(r.text, pipeline.NEEDS_CONTEXT_REPLY)
            self.assertIsNone(r.sql)
            self.assertIsNone(r.rows)
        self.assertEqual(llm.prompts, [])
        self.assertEqual(memory.turn_count("s1"), 0)          # nothing invented, nothing remembered

    def test_dangling_message_after_reset_asks_again(self):
        llm = ScriptedLLM(plan("overview", "total revenue in germany in 2022", SQL_DE_2022))
        self.ask(llm, "total revenue in germany in 2022")
        memory.clear("s1")
        r = self.ask(llm, "what about india")
        self.assertEqual(r.text, pipeline.NEEDS_CONTEXT_REPLY)

    def test_dangling_message_after_ttl_expiry_asks_again(self):
        llm = ScriptedLLM(plan("overview", "total revenue in germany in 2022", SQL_DE_2022))
        self.ask(llm, "total revenue in germany in 2022")
        with memory._lock:
            memory._sessions["s1"].last_seen -= memory.SESSION_TTL_SECONDS + 5
        r = self.ask(llm, "what about india")
        self.assertEqual(r.text, pipeline.NEEDS_CONTEXT_REPLY)

    def test_model_flags_short_followup_without_history(self):
        llm = ScriptedLLM(plan("followup", "only mobile", None))
        r = self.ask(llm, "only mobile")          # not caught by the pre-check (no cue word)
        self.assertEqual(r.text, pipeline.NEEDS_CONTEXT_REPLY)
        self.assertEqual(memory.turn_count("s1"), 0)

    def test_model_guess_is_discarded_for_short_followup_without_history(self):
        llm = ScriptedLLM(plan("followup", "only mobile", "SELECT COUNT(*) AS orders FROM transactions"))
        r = self.ask(llm, "only mobile")
        self.assertEqual(r.text, pipeline.NEEDS_CONTEXT_REPLY)
        self.assertEqual(memory.turn_count("s1"), 0)

    def test_long_real_question_mislabelled_followup_is_still_answered(self):
        q = "total revenue from gold customers in germany during 2022"
        llm = ScriptedLLM(plan("followup", q, SQL_IN_2022_GOLD))
        r = self.ask(llm, q)
        self.assertIsNotNone(r.rows)              # a wrong label must not refuse a good question

    def test_model_cannot_resolve_followup_with_history(self):
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("followup", "unclear", None),
        )
        self.ask(llm, "total revenue in germany in 2022")
        r = self.ask(llm, "why")
        self.assertEqual(r.text, pipeline.UNCLEAR_FOLLOWUP_REPLY)
        self.assertEqual(memory.turn_count("s1"), 1)           # the earlier turn is untouched

    def test_out_of_scope_is_not_remembered_and_keeps_old_context(self):
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("out_of_scope", "weather in paris", None),
        )
        self.ask(llm, "total revenue in germany in 2022")
        r = self.ask(llm, "what is the weather in paris")
        self.assertEqual(r.text, pipeline.OUT_OF_SCOPE_REPLY)
        self.assertEqual(memory.turn_count("s1"), 1)


# --------------------------------------------------------------------------
# What is (and is not) remembered
# --------------------------------------------------------------------------

class WhatIsRemembered(FlowTestCase):
    def test_zero_row_turn_is_remembered_so_the_next_followup_attaches(self):
        """Regression: only turns with rows were remembered, so a follow-up
        after an empty answer lost everything."""
        empty_sql = ("SELECT t.payment_method, SUM(t.order_value) AS revenue FROM transactions t "
                     "WHERE t.transaction_timestamp >= '2019-01-01' AND t.transaction_timestamp < '2020-01-01' "
                     "GROUP BY t.payment_method")
        llm = ScriptedLLM(
            plan("breakdown", "revenue by payment method in 2019", empty_sql),
            plan("breakdown", "revenue by payment method in 2022",
                 empty_sql.replace("2019", "2022").replace("2020", "2023")),
        )
        r1 = self.ask(llm, "revenue by payment method in 2019")
        self.assertEqual(r1.rows, [])
        self.assertEqual(r1.text, pipeline.NO_DATA_REPLY)
        self.assertEqual(memory.turn_count("s1"), 1)
        r2 = self.ask(llm, "what about 2022")
        self.assertIn("no rows (the filters matched nothing)", llm.prompts[1])
        self.assertIn("revenue by payment method in 2019", llm.prompts[1])
        self.assertTrue(len(r2.rows) > 0)

    def test_all_null_aggregate_is_described_as_no_data(self):
        sql = ("SELECT SUM(t.order_value) AS revenue FROM transactions t "
               "WHERE t.transaction_timestamp >= '2010-01-01' AND t.transaction_timestamp < '2011-01-01'")
        llm = ScriptedLLM(plan("overview", "revenue in 2010", sql), plan("overview", "revenue in 2022", SQL_DE_2022))
        self.ask(llm, "revenue in 2010")
        self.ask(llm, "what about 2022")
        self.assertIn("no data (every value was empty)", llm.prompts[1])

    def test_failed_validation_is_not_remembered(self):
        bad = plan("overview", "delete everything", "DELETE FROM customers")
        llm = ScriptedLLM(bad, bad)               # first try + the one retry
        r = self.ask(llm, "delete everything")
        self.assertIn("didn't pass validation", r.text)
        self.assertEqual(memory.turn_count("s1"), 0)

    def test_failed_execution_is_not_remembered(self):
        bad = plan("overview", "revenue", "SELECT SUM(nonexistent_col) FROM transactions")
        llm = ScriptedLLM(bad, bad)
        r = self.ask(llm, "total revenue overall")
        self.assertIn("didn't run cleanly", r.text)
        self.assertEqual(memory.turn_count("s1"), 0)

    def test_retry_keeps_history_and_remembers_the_corrected_turn(self):
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("overview", "total revenue in india in 2022", "DELETE FROM customers"),     # rejected
            plan("overview", "total revenue in india in 2022", SQL_IN_2022),                 # corrected
        )
        self.ask(llm, "total revenue in germany in 2022")
        r = self.ask(llm, "what about india")
        self.assertEqual(r.rows, [{"revenue": 100.0}])
        retry_prompt = llm.prompts[2]
        self.assertIn("Conversation so far:", retry_prompt)            # context survives the retry
        self.assertIn("SQL that ran: " + SQL_DE_2022, retry_prompt)
        self.assertIn("standalone_question", retry_prompt)             # retry asks for the full shape
        self.assertEqual(memory.turn_count("s1"), 2)                   # rejected attempt not stored
        self.assertIn("c.country_code = 'IN'", memory.snapshot("s1")[-1]["sql"])

    def test_value_spelling_is_fixed_before_it_is_remembered(self):
        sloppy = SQL_IN_2022.replace("'IN'", "'india'")
        llm = ScriptedLLM(plan("overview", "total revenue in india in 2022", sloppy))
        r = self.ask(llm, "total revenue in india in 2022")
        self.assertEqual(r.rows, [{"revenue": 100.0}])                 # would be None if 'india' ran as-is
        self.assertIn("'IN'", memory.snapshot("s1")[0]["sql"])

    def test_missing_standalone_question_falls_back_to_typed_question(self):
        llm = ScriptedLLM(plan("overview", "   ", SQL_DE_2022))
        self.ask(llm, "total revenue in germany in 2022")
        self.assertEqual(memory.snapshot("s1")[0]["resolved_question"], "total revenue in germany in 2022")

    def test_multi_row_result_summary_is_informative(self):
        sql = ("SELECT t.payment_method, SUM(t.order_value) AS revenue FROM transactions t "
               "GROUP BY t.payment_method ORDER BY revenue DESC")
        llm = ScriptedLLM(plan("breakdown", "revenue by payment method", sql),
                          plan("overview", "total revenue in germany in 2022", SQL_DE_2022))
        self.ask(llm, "revenue by payment method")
        self.ask(llm, "and germany 2022")
        self.assertIn("4 rows, first: payment_method=debit_card, revenue=200.00", llm.prompts[1])


# --------------------------------------------------------------------------
# Sessions and cache
# --------------------------------------------------------------------------

class SessionsAndCache(FlowTestCase):
    def test_sessions_do_not_leak_into_each_other(self):
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("overview", "how many sessions came from ads",
                 "SELECT COUNT(*) AS sessions FROM sessions WHERE traffic_source = 'ads'"),
        )
        self.ask(llm, "total revenue in germany in 2022", session="A")
        self.ask(llm, "how many sessions came from ads", session="B")
        self.assertNotIn("germany", llm.prompts[1])
        self.assertNotIn("Conversation so far", llm.prompts[1])

    def test_followup_answer_cached_in_one_session_is_not_served_to_another(self):
        llm = ScriptedLLM(plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
                          plan("overview", "total revenue in india in 2022", SQL_IN_2022))
        self.ask(llm, "total revenue in germany in 2022", session="A")
        ra = self.ask(llm, "what about india", session="A")
        self.assertEqual(ra.rows, [{"revenue": 100.0}])
        rb = self.ask(ScriptedLLM(), "what about india", session="B")   # fresh session: nothing to follow
        self.assertEqual(rb.text, pipeline.NEEDS_CONTEXT_REPLY)

    def test_same_first_question_is_cached_across_sessions(self):
        llm = ScriptedLLM(plan("overview", "total revenue in germany in 2022", SQL_DE_2022))
        self.ask(llm, "total revenue in germany in 2022", session="A")
        r = self.ask(ScriptedLLM(), "total revenue in germany in 2022", session="B")   # would raise if the model were called
        self.assertEqual(r.rows, [{"revenue": 150.0}])
        self.assertEqual(memory.turn_count("B"), 1)          # cache hits are still remembered

    def test_asking_the_same_thing_twice_does_not_duplicate_history(self):
        llm = ScriptedLLM(*[plan("overview", "total revenue in germany in 2022", SQL_DE_2022)] * 3)
        for _ in range(3):
            self.ask(llm, "total revenue in germany in 2022")
        self.assertEqual(memory.turn_count("s1"), 1)

    def test_no_session_id_works_and_remembers_nothing(self):
        llm = ScriptedLLM(plan("overview", "total revenue in germany in 2022", SQL_DE_2022))
        with mock.patch.object(pipeline.llm, "generate_structured", llm):
            r = pipeline.answer_question("total revenue in germany in 2022", session_id=None)
        self.assertEqual(r.rows, [{"revenue": 150.0}])


class Concurrency(FlowTestCase):
    def test_quick_second_message_sees_the_first_one(self):
        """Regression: two messages in one session used to be processed in
        parallel, so the second was built without the first as context."""
        llm = ScriptedLLM(
            plan("overview", "total revenue in germany in 2022", SQL_DE_2022),
            plan("overview", "total revenue in india in 2022", SQL_IN_2022),
            delay=0.25,
        )
        results = {}

        def run(name, q, wait):
            time.sleep(wait)
            results[name] = pipeline.answer_question(q, session_id="s1")

        t1 = threading.Thread(target=run, args=("first", "total revenue in germany in 2022", 0))
        t2 = threading.Thread(target=run, args=("second", "what about india", 0.05))
        with mock.patch.object(pipeline.llm, "generate_structured", llm):   # patched once, not per thread
            t1.start(); t2.start(); t1.join(); t2.join()
        self.assertIn("User typed: total revenue in germany in 2022", llm.prompts[1])
        self.assertEqual(results["second"].rows, [{"revenue": 100.0}])


# --------------------------------------------------------------------------
# Standard (non-fast) mode
# --------------------------------------------------------------------------

class StandardMode(FlowTestCase):
    def test_standalone_question_flows_into_memory_in_standard_mode(self):
        from chatbot.intent_schema import ExtractedQuery
        from chatbot.sql_schema import GeneratedSQL

        script = [
            ExtractedQuery(intent=Intent.OVERVIEW, standalone_question="total revenue in germany in 2022"),
            GeneratedSQL(sql=SQL_DE_2022),
            ExtractedQuery(intent=Intent.OVERVIEW, standalone_question="total revenue in india in 2022"),
            GeneratedSQL(sql=SQL_IN_2022),
        ]
        prompts = []

        def fake(system, prompt, schema, max_tokens=300):
            prompts.append(prompt)
            return script.pop(0)

        with mock.patch.object(pipeline, "FAST_MODE", False), \
             mock.patch.object(pipeline.llm, "generate_structured", fake):
            pipeline.answer_question("total revenue in germany in 2022", session_id="s1")
            r2 = pipeline.answer_question("what about india", session_id="s1")
        self.assertEqual(r2.rows, [{"revenue": 100.0}])
        self.assertEqual(r2.interpreted_as, "total revenue in india in 2022")
        self.assertIn("User typed: total revenue in germany in 2022", prompts[2])   # intent call
        self.assertIn("Conversation so far", prompts[3])                            # SQL call
        self.assertEqual(memory.snapshot("s1")[-1]["resolved_question"], "total revenue in india in 2022")


if __name__ == "__main__":
    unittest.main()
