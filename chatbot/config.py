"""
config.py

All the knobs the pipeline needs, each overridable with an environment
variable so the same code runs against a different Ollama host, model,
or database path without editing source.
"""
import os
from pathlib import Path


def _opt_int(name: str):
    """Optional int env var: unset/empty -> None (let Ollama decide)."""
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


# --- Ollama ---------------------------------------------------------------
# "localhost" is swapped for 127.0.0.1: on Windows, resolving "localhost" (IPv6 ::1
# first) cost ~2.07 s on EVERY new connection in the Langfuse traces of this project.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").replace("//localhost", "//127.0.0.1")
# httpx closes idle connections after 5 s by default, so between two questions
# (always > 5 s apart) every call reconnected. Keep the connection for this long.
HTTP_KEEPALIVE_S = float(os.environ.get("CHATBOT_HTTP_KEEPALIVE_S", "900"))
# Qwen2.5-Coder is a non-"thinking" model that is strong at SQL. For a faster,
# slightly less accurate option use qwen2.5-coder:3b.
LLM_MODEL = os.environ.get("CHATBOT_LLM_MODEL", "qwen2.5-coder:3b")
LLM_KEEP_ALIVE = os.environ.get("CHATBOT_LLM_KEEP_ALIVE", "30m")
# 1 is right for an iGPU: every extra parallel slot multiplies the KV cache
# and breaks prompt-prefix reuse, which is the biggest speed win here.
LLM_NUM_PARALLEL = int(os.environ.get("CHATBOT_LLM_NUM_PARALLEL", "1"))

# Context window. MUST be set explicitly: Ollama's default (2k-4k depending on
# version) can silently truncate the schema prompt. Keep it constant -- changing
# it between calls makes Ollama reload the model.
LLM_NUM_CTX = int(os.environ.get("CHATBOT_LLM_NUM_CTX", "6144"))
# Layers to put on the GPU. Unset = Ollama decides. 999 = force everything on
# the GPU (try this if `check_model` shows a partial offload).
LLM_NUM_GPU = _opt_int("CHATBOT_LLM_NUM_GPU")
# CPU threads for whatever isn't on the GPU. Unset = Ollama decides. On a
# Ryzen 7 8845HS (8 cores / 16 threads) 8 is a good value to try.
LLM_NUM_THREAD = _opt_int("CHATBOT_LLM_NUM_THREAD")
# "auto": turn thinking off for Qwen3-style reasoning models (it wastes tokens
# and fights JSON-constrained output). "off"/"on" force it.
LLM_THINK = os.environ.get("CHATBOT_LLM_THINK", "auto").lower()

# --- Token budgets --------------------------------------------------------
INTENT_MAX_TOKENS = int(os.environ.get("CHATBOT_INTENT_MAX_TOKENS", "220"))
SQL_MAX_TOKENS = int(os.environ.get("CHATBOT_SQL_MAX_TOKENS", "256"))
NL_MAX_TOKENS = int(os.environ.get("CHATBOT_NL_MAX_TOKENS", "200"))

# --- Database -------------------------------------------------------------
DB_PATH = Path(
    os.environ.get(
        "ECOMMERCE_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "sqlite_db" / "ecommerce.db"),
    )
)
MAX_ROWS_RETURNED = int(os.environ.get("CHATBOT_MAX_ROWS", "200"))
REFERENCE_DATE_TTL_S = int(os.environ.get("CHATBOT_REFERENCE_DATE_TTL_S", "300"))

# --- HTTP / LLM timeouts --------------------------------------------------
# Generous: the very first call after a cold start has to load the model.
REQUEST_TIMEOUT_S = int(os.environ.get("CHATBOT_LLM_TIMEOUT_S", "120"))
SQL_QUERY_TIMEOUT_S = float(os.environ.get("CHATBOT_SQL_TIMEOUT_S", "15"))

# --- Caching --------------------------------------------------------------
CACHE_MAX_ENTRIES = int(os.environ.get("CHATBOT_CACHE_MAX_ENTRIES", "500"))
CACHE_TTL_S = int(os.environ.get("CHATBOT_CACHE_TTL_S", "900"))

# --- Pipeline behaviour ---------------------------------------------------
FAST_MODE = os.environ.get("CHATBOT_FAST_MODE", "1") == "1"
MAX_RETRIES = int(os.environ.get("CHATBOT_MAX_RETRIES", "1"))
MAX_LISTED_ROWS = int(os.environ.get("CHATBOT_MAX_LISTED_ROWS", "12"))

# --- Follow-up memory -----------------------------------------------------
# Turns kept per session, sessions kept in total, idle time before a session
# is forgotten, and the max size (characters) of the history block that is put
# in the prompt.
#
# MEMORY_CONTEXT_CHARS is a speed knob: the history sits in the user message,
# so Ollama cannot cache it and re-reads it on every follow-up at the iGPU's
# ~65 tok/s prompt speed (~3 chars per token for SQL-heavy text):
#     3500 chars -> up to ~1,100 tokens -> ~17 s of pure reading (the old value)
#     1200 chars ->  up to  ~400 tokens ->  ~6 s
# Measured on realistic turns, 1200 holds the last three turns (the newest
# one is always kept in full, whatever the budget). Raise it only if
# follow-ups lose context; lower it (e.g. 800 = newest turn only) for speed.
MEMORY_MAX_TURNS = int(os.environ.get("CHATBOT_MEMORY_TURNS", "6"))
MEMORY_MAX_SESSIONS = int(os.environ.get("CHATBOT_MEMORY_SESSIONS", "500"))
MEMORY_TTL_S = int(os.environ.get("CHATBOT_MEMORY_TTL_S", str(60 * 60)))
MEMORY_CONTEXT_CHARS = int(os.environ.get("CHATBOT_MEMORY_CONTEXT_CHARS", "1200"))
# Max seconds a request waits for an earlier request in the SAME session.
MEMORY_TURN_WAIT_S = int(os.environ.get("CHATBOT_MEMORY_TURN_WAIT_S", "150"))
