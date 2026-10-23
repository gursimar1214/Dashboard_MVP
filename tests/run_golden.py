"""
run_golden.py -- runs tests/golden_set.json through the REAL pipeline.

Run from the project root, with Ollama running:

    python tests/run_golden.py                 # all 45 questions
    python tests/run_golden.py --level 3       # one difficulty level
    python tests/run_golden.py --only G06 G17  # specific questions

Compare models or modes by re-running with different settings, e.g.
    $env:CHATBOT_LLM_MODEL = "qwen2.5-coder:3b"
    $env:CHATBOT_FAST_MODE = "0"
Each run saves tests/results_<model>_<mode>.json so you can put the
numbers side by side.

How answers are judged (against values computed by hand-written SQL on
the real database -- see "reference_sql" in golden_set.json):
  scalar : exactly one row, containing the expected number
  table  : one row per expected label, each with the expected number
  top    : the first rows match the expected values (and labels, if unique)
  zero   : no rows, or a single 0  (the honest "nothing there" answer)
  refuse : no data returned at all (out-of-scope reply or a blocked query)
Rates may be given as a fraction (0.4995) or a percent (49.95).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chatbot import pipeline  # noqa: E402
from chatbot.config import LLM_MODEL  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden_set.json"
LEVEL_NAMES = {1: "basics", 2: "exact values", 3: "joins", 4: "group/rank/time",
               5: "extreme", 6: "traps & refusals"}


def close(a: float, b: float) -> bool:
    tol = 0.006 if abs(b) >= 1 else 0.0006
    return abs(a - b) <= max(tol, abs(b) * 1e-9)


def row_numbers(row: dict) -> list[float]:
    return [float(v) for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]


def row_labels(row: dict) -> set[str]:
    return {str(v).strip().lower() for v in row.values() if v is not None}


def has_number(row: dict, targets: list[float]) -> bool:
    return any(close(n, t) for n in row_numbers(row) for t in targets)


def judge(case: dict, answer) -> tuple[bool, str]:
    rows, kind = answer.rows, case["kind"]
    if kind == "refuse":
        ok = rows is None or answer.intent == "out_of_scope"
        return ok, "" if ok else f"returned {len(rows)} row(s) instead of refusing"
    if kind == "zero":
        if rows == []:
            return True, ""
        ok = rows is not None and len(rows) == 1 and all(v in (0, None) for v in rows[0].values())
        return ok, "" if ok else "expected 0 / no data"
    if rows is None:
        return False, "no rows (error or refusal)"
    if kind == "scalar":
        if len(rows) != 1:
            return False, f"expected 1 row, got {len(rows)}"
        ok = has_number(rows[0], case["accept"])
        return ok, "" if ok else f"expected {case['expected']}"
    if kind == "table":
        exp = case["expected"]
        if len(rows) != len(exp):
            return False, f"expected {len(exp)} rows, got {len(rows)}"
        for label, value in exp.items():
            targets = [value] + ([value * 100] if case.get("pct_ok") else [])
            if not any(label.lower() in row_labels(r) and has_number(r, targets) for r in rows):
                return False, f"missing/incorrect row for '{label}' (expected {value})"
        return True, ""
    if kind == "top":
        exp = case["expected"]
        if len(rows) != len(exp):
            return False, f"expected {len(exp)} row(s), got {len(rows)}"
        for i, e in enumerate(exp):
            if not has_number(rows[i], [e["value"]]):
                return False, f"row {i + 1}: expected value {e['value']}"
            if e.get("label") and e["label"].lower() not in row_labels(rows[i]):
                return False, f"row {i + 1}: expected '{e['label']}'"
        return True, ""
    return False, f"unknown kind {kind}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if args.level:
        cases = [c for c in cases if c["level"] == args.level]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    mode = "fast" if pipeline.FAST_MODE else "standard"
    print(f"Model: {LLM_MODEL} | mode: {mode} | questions: {len(cases)}\n")

    results, by_level = [], {}
    for case in cases:
        start = time.perf_counter()
        try:
            answer = pipeline.answer_question(case["question"])
            ok, why = judge(case, answer)
            sql, reply = answer.sql, answer.text
        except Exception as exc:  # noqa: BLE001
            ok, why, sql, reply = False, f"exception: {exc}", None, ""
        secs = round(time.perf_counter() - start, 1)
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']}  L{case['level']}  {secs:>5}s  {case['question']}")
        if not ok:
            print(f"        why : {why}")
            print(f"        sql : {sql}")
            print(f"        said: {reply[:150]!r}")
        stat = by_level.setdefault(case["level"], [0, 0])
        stat[0] += ok
        stat[1] += 1
        results.append({"id": case["id"], "level": case["level"], "question": case["question"],
                        "passed": ok, "why": why, "sql": sql, "reply": reply, "seconds": secs})

    total = sum(r["passed"] for r in results)
    print("\n" + "=" * 60)
    for lvl in sorted(by_level):
        p, n = by_level[lvl]
        print(f"Level {lvl} ({LEVEL_NAMES.get(lvl, '')}): {p}/{n}")
    times = sorted(r["seconds"] for r in results)
    if results:
        print(f"OVERALL: {total}/{len(results)} = {100 * total / len(results):.0f}%")
        print(f"Latency: avg {sum(times) / len(times):.1f}s | slowest {times[-1]}s")
    out = Path(__file__).resolve().parent / f"results_{LLM_MODEL.replace(':', '-').replace('/', '-')}_{mode}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    main()
