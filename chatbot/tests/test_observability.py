"""
Tests for the Langfuse layer. No Ollama, no network, no langfuse install
(a fake `langfuse` module is injected).

    python -m unittest chatbot.tests.test_observability -v   (from the project root)
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
import types
import unittest

KEYS = {"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b"}
_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL",
        "CHATBOT_LANGFUSE_FULL_PROMPT")


def _fake_langfuse(calls: list, auth_ok: bool = True, with_propagate: bool = True) -> types.ModuleType:
    mod = types.ModuleType("langfuse")

    class _Span:
        def __init__(self, name): self.name = name
        def update(self, **k): calls.append(("update", self.name, sorted(k)))

    class _CM:
        def __init__(self, name): self.name = name
        def __enter__(self): calls.append(("enter", self.name)); return _Span(self.name)
        def __exit__(self, et, ev, tb): calls.append(("exit", self.name, et.__name__ if et else None)); return False

    class _Client:
        def start_as_current_observation(self, **k): return _CM(k["name"])
        def get_current_trace_id(self): return "abc123"
        def auth_check(self): return auth_ok
        def update_current_trace(self, **k): calls.append(("update_current_trace", k))
        def create_score(self, **k): calls.append(("score", k["name"], k["value"], k["data_type"], k.get("score_id")))
        def flush(self): calls.append(("flush",))
        def shutdown(self): calls.append(("shutdown",))

    mod.get_client = lambda: _Client()
    if with_propagate:
        @contextlib.contextmanager
        def propagate_attributes(**k):
            calls.append(("propagate", k)); yield
        mod.propagate_attributes = propagate_attributes
    return mod


def _fresh_obs(env: dict, fake):
    for k in _ENV:
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop("langfuse", None)
    if fake is not None:
        sys.modules["langfuse"] = fake
    from .. import observability
    return importlib.reload(observability)


class ObservabilityTests(unittest.TestCase):
    def tearDown(self):
        for k in _ENV:
            os.environ.pop(k, None)
        sys.modules.pop("langfuse", None)

    def _on(self, **kw):
        calls: list = []
        obs = _fresh_obs(KEYS, _fake_langfuse(calls, **kw))
        self.assertTrue(obs.init())
        return obs, calls

    # ---- off ----------------------------------------------------------------
    def test_off_without_keys_is_a_silent_noop(self):
        obs = _fresh_obs({}, None)
        self.assertFalse(obs.init())
        with obs.request_trace("r", "s", "q") as root:
            with obs.observation("child") as c:
                c.update(output=1)
            root.update(output="x")
        self.assertIsNone(root.trace_id)
        self.assertFalse(obs.score("abc", "n", 1))
        obs.shutdown()

    def test_off_when_package_missing_or_auth_fails(self):
        self.assertFalse(_fresh_obs(KEYS, None).init())
        self.assertFalse(_fresh_obs(KEYS, _fake_langfuse([], auth_ok=False)).init())

    def test_either_host_variable_is_accepted(self):
        obs = _fresh_obs({**KEYS, "LANGFUSE_HOST": "http://localhost:3000"}, _fake_langfuse([]))
        obs.init()
        self.assertEqual(os.environ["LANGFUSE_BASE_URL"], "http://localhost:3000")

    # ---- on -------------------------------------------------------------------
    def test_session_and_tags_use_propagate_attributes_around_the_root(self):
        obs, calls = self._on()
        with obs.request_trace("chat-request", "sess1", "q", tags=["model:m", "mode:fast"]) as root:
            with obs.observation("sql-plan", as_type="generation", model="m") as g:
                g.update(output="o")
            self.assertEqual(root.trace_id, "abc123")
        self.assertEqual(calls[0], ("propagate", {"session_id": "sess1", "tags": ["model:m", "mode:fast"]}))
        order = [c[:2] for c in calls if c[0] in ("enter", "exit")]
        self.assertEqual(order, [("enter", "chat-request"), ("enter", "sql-plan"),
                                 ("exit", "sql-plan"), ("exit", "chat-request")])
        self.assertFalse([c for c in calls if c[0] == "update_current_trace"])   # not the v3-only call

    def test_older_v3_without_propagate_falls_back_to_update_current_trace(self):
        obs, calls = self._on(with_propagate=False)
        with obs.request_trace("chat-request", "sess1", "q", tags=["t"]):
            pass
        self.assertIn(("update_current_trace", {"session_id": "sess1", "tags": ["t"]}), calls)

    def test_exception_in_the_body_propagates_unchanged(self):
        """Regression: request_trace used to catch it and yield a second time,
        replacing the real error with 'generator didn't stop after throw()'."""
        obs, calls = self._on()
        with self.assertRaises(ValueError) as cm:
            with obs.request_trace("chat-request", "s", "q"):
                raise ValueError("pipeline blew up")
        self.assertEqual(str(cm.exception), "pipeline blew up")
        self.assertIn(("update", "chat-request", ["level", "status_message"]), calls)
        self.assertIn(("exit", "chat-request", "ValueError"), calls)

    def test_child_exception_marks_error_and_propagates(self):
        obs, calls = self._on()
        with self.assertRaises(ValueError):
            with obs.observation("validate"):
                raise ValueError("boom")
        self.assertIn(("update", "validate", ["level", "status_message"]), calls)

    def test_numeric_and_categorical_scores_are_idempotent(self):
        obs, calls = self._on()
        self.assertTrue(obs.score("abc123", "user_feedback", 1))
        self.assertTrue(obs.score("abc123", "user_feedback", 0, comment="wrong filter"))
        self.assertTrue(obs.score("abc123", "path", "llm"))
        scores = [c for c in calls if c[0] == "score"]
        self.assertEqual({c[4] for c in scores if c[1] == "user_feedback"}, {"abc123-user_feedback"})
        self.assertIn(("score", "user_feedback", 1.0, "NUMERIC", "abc123-user_feedback"), scores)
        self.assertIn(("score", "path", "llm", "CATEGORICAL", "abc123-path"), scores)

    def test_shutdown_flushes(self):
        obs, calls = self._on()
        obs.shutdown()
        self.assertIn(("flush",), calls)

    # ---- payload size ------------------------------------------------------------
    def test_system_prompt_is_fingerprinted_not_sent(self):
        obs = _fresh_obs({}, None)
        msgs = [{"role": "system", "content": "x" * 9000}, {"role": "user", "content": "Question: hi"}]
        out = obs.prepare_messages(msgs)
        self.assertEqual(out[1], msgs[1])
        self.assertIn("9000 chars", out[0]["content"])
        self.assertLess(len(out[0]["content"]), 200)

    def test_full_prompt_switch(self):
        obs = _fresh_obs({"CHATBOT_LANGFUSE_FULL_PROMPT": "1"}, None)
        msgs = [{"role": "system", "content": "x" * 50}]
        self.assertEqual(obs.prepare_messages(msgs), msgs)

    def test_trim_rows_limits_what_leaves_the_machine(self):
        obs = _fresh_obs({}, None)
        t = obs.trim_rows([{"i": i} for i in range(50)])
        self.assertEqual((t["row_count"], len(t["sample_rows"])), (50, 5))
        self.assertIsNone(obs.trim_rows(None))


if __name__ == "__main__":
    unittest.main()
