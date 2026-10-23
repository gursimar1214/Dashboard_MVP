"""Tests added with the audit fixes. Run from the project root (the folder
that contains chatbot/):

    python -m unittest chatbot.tests.test_audit_fixes -v     # no Ollama needed
    python -m chatbot.tests.audit_questions --dry            # checks ground truth only
    python -m chatbot.tests.audit_questions                  # needs Ollama running

They live inside the package on purpose, so they never collide with a
project-level tests/ folder (test_units.py, run_golden.py).
"""
