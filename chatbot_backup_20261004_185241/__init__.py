"""Local, fully-on-machine chatbot for the Global Ecommerce Dashboard.

Pipeline (7 stages):
  1. User message         -> FastAPI            (main.py)
  2. Intent & entities     -> Qwen2.5-Coder (Ollama) + structured decoding  (intent_schema.py, llm.py)
  3. Prompt assembly       -> Python (schema + metrics glossary)          (schema_glossary.py, pipeline.py)
  4. SQL generation        -> Qwen2.5-Coder (Ollama) + grammar-constrained decoding (sql_schema.py, llm.py)
  5. SQL validation         -> sqlglot                                     (sql_validator.py)
  6. Query execution        -> sqlite3, read-only                          (db.py)
  7. Natural language reply -> Qwen2.5-Coder (Ollama)                       (llm.py)

See README.md in this folder for setup and an architecture diagram.
"""