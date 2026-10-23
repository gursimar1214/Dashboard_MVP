"""
check_model.py -- is Qwen installed, answering, and running on the GPU?

Run from the project root (the folder that contains chatbot/):

    python -m chatbot.check_model              # model + speed + GPU check
    python -m chatbot.check_model --pipeline   # also run 3 real questions end to end

Exit code 0 = model works, 1 = something is wrong (the reason is printed).
"""
from __future__ import annotations

import argparse
import sys
import time

from . import llm
from .config import LLM_MODEL, LLM_NUM_CTX, LLM_NUM_GPU, OLLAMA_HOST


def _ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _speed(count, duration_ns) -> str:
    if not count or not duration_ns:
        return "n/a"
    return f"{count / (duration_ns / 1e9):.1f} tok/s"


def check_server_and_model() -> bool:
    print(f"\n1) Ollama + model   (host {OLLAMA_HOST}, model {LLM_MODEL})")
    status = llm.model_status()
    if not status["reachable"]:
        _fail("Can't reach Ollama. Start it (`ollama serve`, or open the Ollama app).")
        return False
    _ok("Ollama is running.")
    if not status["installed"]:
        _fail(f"Model not installed. Run:  ollama pull {LLM_MODEL}")
        return False
    _ok(f"Model '{LLM_MODEL}' is installed.")
    return True


def check_generation() -> bool:
    print("\n2) Real generation (JSON-constrained, same call type the chatbot uses)")
    from pydantic import BaseModel

    class Probe(BaseModel):
        answer: int

    started = time.perf_counter()
    try:
        response = llm._chat(
            [
                {"role": "system", "content": "Reply with JSON only."},
                {"role": "user", "content": "What is 17 + 25? Put the number in the field 'answer'."},
            ],
            llm._options(0, 40),
            format=Probe.model_json_schema(),
        )
    except Exception as exc:
        err = llm._translate(exc)
        _fail(err.user_message)
        return False
    wall = time.perf_counter() - started

    content = llm._get(llm._get(response, "message"), "content", "")
    try:
        value = Probe.model_validate_json(content).answer
    except Exception:
        _fail(f"Model replied, but not with valid JSON: {content!r}")
        return False
    (_ok if value == 42 else _warn)(f"Model replied {value} (expected 42) in {wall:.1f}s total.")

    load_s = (llm._get(response, "load_duration", 0) or 0) / 1e9
    print(f"         model load time : {load_s:.1f}s (first call only)")
    print(
        "         prompt speed    : "
        + _speed(llm._get(response, "prompt_eval_count"), llm._get(response, "prompt_eval_duration"))
    )
    print(
        "         output speed    : "
        + _speed(llm._get(response, "eval_count"), llm._get(response, "eval_duration"))
    )
    return True


def check_gpu() -> None:
    print("\n3) GPU usage")
    status = llm.model_status()
    if not status["loaded"]:
        _warn("Model isn't loaded right now, so GPU share can't be read.")
        return
    gpu = status["gpu_percent"]
    size_gb = (status.get("size_bytes") or 0) / 1e9
    vram_gb = (status.get("vram_bytes") or 0) / 1e9
    print(f"         model size {size_gb:.1f} GB, of which {vram_gb:.1f} GB on the GPU")
    if gpu is None:
        _warn("Ollama didn't report a GPU share.")
    elif gpu >= 95:
        _ok(f"{gpu}% of the model is on the GPU.")
    elif gpu > 0:
        _warn(
            f"Only {gpu}% is on the GPU; the rest runs on CPU. Try "
            f"CHATBOT_LLM_NUM_GPU=999, a smaller model (qwen2.5-coder:3b), or a smaller "
            f"CHATBOT_LLM_NUM_CTX."
        )
    else:
        _fail(
            "0% on GPU -- Ollama is using the CPU only.\n"
            "         Ryzen 8845HS's Radeon 780M isn't supported by ROCm, so it needs Ollama's\n"
            "         Vulkan backend. Windows: run chatbot\\run_ollama_gpu.ps1 . Linux: set\n"
            "         OLLAMA_VULKAN=1 for the Ollama service (see README), then restart Ollama.\n"
            "         Also make sure your AMD graphics driver is up to date."
        )
    if LLM_NUM_GPU is not None:
        print(f"         (CHATBOT_LLM_NUM_GPU is set to {LLM_NUM_GPU})")
    print(f"         (context window: {LLM_NUM_CTX} tokens)")


def check_pipeline() -> bool:
    print("\n4) End-to-end pipeline (needs the SQLite database)")
    from .pipeline import answer_question, warmup

    try:
        warmup()
    except llm.LLMError as exc:
        _fail(exc.user_message)
        return False

    all_good = True
    for question in (
        "how many customers are there",
        "which payment method brings in the most revenue",
        "average order value for gold customers in germany",
    ):
        t = time.perf_counter()
        result = answer_question(question)
        secs = time.perf_counter() - t
        good = result.rows is not None and result.intent != "error"
        all_good &= good
        (_ok if good else _fail)(f"{question!r}  ({secs:.1f}s)")
        print(f"         SQL   : {result.sql}")
        print(f"         answer: {result.text.splitlines()[0] if result.text else ''}")
    return all_good


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local Qwen model and GPU use.")
    parser.add_argument("--pipeline", action="store_true", help="also run 3 real questions")
    args = parser.parse_args()

    if not check_server_and_model():
        return 1
    if not check_generation():
        return 1
    check_gpu()
    if args.pipeline and not check_pipeline():
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
