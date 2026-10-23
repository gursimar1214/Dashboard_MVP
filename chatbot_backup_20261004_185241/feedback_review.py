"""
feedback_review.py -- turn thumbs-down answers into things to fix.

    python -m chatbot.feedback_review                # list thumbs-down answers
    python -m chatbot.feedback_review --all          # include thumbs-up too
    python -m chatbot.feedback_review --write-tests  # also draft test cases

--write-tests appends one line per thumbs-down answer to
chatbot/tests/feedback_cases.jsonl:
    {"id", "q", "bad_sql", "comment", "expected_sql": null, "status": "todo"}
Fill in `expected_sql` (the query you KNOW is right), then copy the case into
CASES in tests/audit_questions.py as  dict(id=..., q=..., gt=<expected_sql>).
After you fix the glossary / synonyms, `python -m chatbot.tests.audit_questions`
proves the question now works and keeps it working.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import feedback

CASES_FILE = Path(__file__).parent / "tests" / "feedback_cases.jsonl"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "case"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="include thumbs-up answers")
    ap.add_argument("--write-tests", action="store_true", help="append draft cases to tests/feedback_cases.jsonl")
    ap.add_argument("--log", help="feedback log path (default: the configured one)")
    args = ap.parse_args()

    entries = feedback.latest_per_answer(feedback.read_log(Path(args.log) if args.log else None))
    shown = [e for e in entries if args.all or not e.get("helpful", True)]
    if not shown:
        print("No thumbs-down feedback yet." if not args.all else "No feedback yet.")
        return

    existing: set[str] = set()
    if CASES_FILE.exists():
        existing = {json.loads(l).get("id", "") for l in CASES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}

    drafted = 0
    for e in shown:
        mark = "UP  " if e.get("helpful") else "DOWN"
        print(f"\n[{mark}] {e.get('time')}  path={e.get('path')}  resolution={e.get('resolution')}  model={e.get('model')}")
        print(f"  asked      : {e.get('question')}")
        if e.get("understood"):
            print(f"  understood : {e.get('understood')}")
        print(f"  sql        : {e.get('sql')}")
        print(f"  answer     : {e.get('answer')}")
        if e.get("comment"):
            print(f"  note       : {e.get('comment')}")
        if args.write_tests and not e.get("helpful", True):
            cid = f"fb_{_slug(e.get('question') or '')}_{(e.get('answer_id') or '')[:6]}"
            if cid not in existing:
                CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
                with CASES_FILE.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "id": cid, "q": e.get("question"), "bad_sql": e.get("sql"),
                        "comment": e.get("comment"), "expected_sql": None, "status": "todo",
                    }, ensure_ascii=False) + "\n")
                drafted += 1
    if args.write_tests:
        print(f"\nDrafted {drafted} new case(s) in {CASES_FILE}")


if __name__ == "__main__":
    main()
