#!/usr/bin/env python3
"""
Test assistant-turn prefills against RunPod vLLM endpoints.

HOW IT WORKS
------------
Place a partial assistant message at the end of the messages array, then send
two vLLM-specific parameters so the server continues generation from that
partial turn rather than starting fresh:

    extra_body={
        "continue_final_message": True,   # continue from the partial turn
        "add_generation_prompt": False,    # don't re-inject <|im_start|>assistant
    }

Without add_generation_prompt=False the chat template appends a fresh
generation prompt, making the context incoherent and causing immediate EOS.

PREFILL RULES FOR QWEN3
-----------------------
- Open <think> block  -> model must close it, then generate response
  e.g. "<think> think something cheeky."
- Closed <think></think> block -> model sees response as complete, hits EOS
- </think>\n only   -> model skips thinking, jumps straight to answer
- Plain text        -> model continues the sentence (must be clearly incomplete)

DEFAULT EXAMPLE (runs when no prompt is given)
----------------------------------------------
    python run_experiments_prefills.py -m qwen3-8b --json-out results/prefill_demo.json

    Prompt:  "Hello there!"
    Battery: no_prefill | cheeky_think | think_close | think_seeded | answer_steer

Usage:
    # Run full default battery, save prefill_demo.json
    python run_experiments_prefills.py -m qwen3-8b --json-out results/prefill_demo.json

    # Single custom prefill
    python run_experiments_prefills.py -m qwen3-8b --prefill "</think>" "Explain gravity"

    # Custom prompt + battery
    python run_experiments_prefills.py -m qwen3-32b --json-out results/out.json "What is 2+2?"
"""

import argparse
import json
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_experiment import (
    load_endpoints,
    build_client,
    _parse_thinking,
    _count_local_tokens,
    _p,
)
import httpx
from openai import APITimeoutError, APIConnectionError, InternalServerError, APIStatusError

SCRIPT_DIR = Path(__file__).parent

DEFAULT_PROMPT = "Hello there!"

DEFAULT_PREFILLS: list[dict] = [
    {
        "label": "no_prefill",
        "text": None,
        "description": "Baseline - no assistant prefill, fresh generation",
    },
    {
        "label": "cheeky_think",
        "text": "<think> I really hate this guy omg, let's say hi and be cheeky.",
        "description": "Open <think> block seeds a hidden thought; model closes it then generates a response",
    },
    {
        "label": "think_close",
        "text": "</think>\n",
        "description": "Prefill </think> to skip thinking entirely and jump straight to the answer",
    },
    {
        "label": "think_seeded",
        "text": "<think>\nLet me think step by step.\n",
        "description": "Seed the opening line of the think block",
    },
    {
        "label": "answer_steer",
        "text": "The answer is ",
        "description": "Steer directly toward an answer with no think block",
    },
]


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _build_messages(
    prompt: str,
    system_prompt: str | None,
    prefill: str | None,
) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
    return messages


def _send(
    client,
    model_name: str,
    messages: list[dict],
    *,
    max_tokens: int | None,
    temperature: float,
    seed: int | None,
    debug: bool,
) -> dict:
    """Send a streaming chat completion, applying prefill params when needed."""
    params = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
    )
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if seed is not None:
        params["seed"] = seed

    has_prefill = messages and messages[-1]["role"] == "assistant"
    if has_prefill:
        params["extra_body"] = {
            "continue_final_message": True,
            "add_generation_prompt": False,
        }

    t_start = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    stop_hb = threading.Event()

    def _hb():
        while not stop_hb.is_set():
            if stop_hb.wait(10):
                break
            print(f"  ... still waiting ({time.perf_counter() - t_start:.0f}s elapsed)", flush=True)

    hb = threading.Thread(target=_hb, daemon=True)
    hb.start()

    try:
        stream = client.chat.completions.create(**params)
        collected = []
        first_tok = None
        chunks = 0
        usage = None

        for i, chunk in enumerate(stream):
            now = time.perf_counter()
            if debug and i < 5:
                _p(f"  [DEBUG chunk {i}] {chunk}")
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if first_tok is None:
                    first_tok = now
                    print(f"[TTFT: {first_tok - t_start:.2f}s] ", end="", flush=True)
                print(delta.content, end="", flush=True)
                collected.append(delta.content)
                chunks += 1

    except (APITimeoutError, APIConnectionError, InternalServerError, APIStatusError,
            httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
        return {"error": str(e), "timestamp_utc": timestamp}
    finally:
        stop_hb.set()
        hb.join(timeout=1)

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    generated = "".join(collected)
    print()

    _p(f"\n{'-' * 60}")
    _p(f"  Total time:          {elapsed:.2f}s")
    if first_tok:
        _p(f"  Time to first token: {first_tok - t_start:.2f}s")
        gen_time = t_end - first_tok
        if chunks > 1:
            _p(f"  Throughput:          ~{chunks / gen_time:.1f} chunks/s")
    if usage:
        _p(f"  Prompt tokens:       {usage.prompt_tokens}")
        _p(f"  Completion tokens:   {usage.completion_tokens}")
        _p(f"  Total tokens:        {usage.total_tokens}")
    _p(f"{'-' * 60}\n")

    return {
        "generated_text": generated,
        "elapsed_s": elapsed,
        "ttft_s": (first_tok - t_start) if first_tok else None,
        "stream_chunks": chunks,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        } if usage else None,
        "timestamp_utc": timestamp,
    }


# ---------------------------------------------------------------------------
# Battery runner
# ---------------------------------------------------------------------------

def run_prefill_battery(
    client,
    model_name: str,
    prompt: str,
    prefills: list[dict],
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    seed: int | None = None,
    debug: bool = False,
) -> list[dict]:
    results = []

    for i, pf in enumerate(prefills):
        label = pf["label"]
        prefill_text = pf["text"]
        description = pf.get("description", "")

        _p(f"\n{'=' * 60}")
        _p(f"  [{i + 1}/{len(prefills)}] {label}")
        _p(f"  {description}")
        _p(f"  prefill: {repr(prefill_text)}")
        _p(f"{'=' * 60}")

        messages = _build_messages(prompt, system_prompt, prefill_text)
        raw = _send(
            client, model_name, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            debug=debug,
        )

        if "error" in raw:
            _p(f"  ERROR: {raw['error']}")
            results.append({"label": label, "description": description,
                            "prefill_text": prefill_text, "error": raw["error"]})
            continue

        # Reconstruct full output: prefill + generated (server only returns the continuation)
        full = (prefill_text or "") + raw["generated_text"]
        thinking_content, response_content = _parse_thinking(full)

        # Local token counts (prompt counted without the prefill stub)
        local_counts = _count_local_tokens(
            _build_messages(prompt, system_prompt, None),
            thinking_content,
            response_content,
            model_name,
        )
        if local_counts:
            su = raw.get("usage") or {}
            _p(f"  [local token counts]")
            _p(f"    prompt:           {local_counts['prompt']}  (server: {su.get('prompt_tokens', 'n/a')})")
            _p(f"    thinking:         {local_counts['thinking']}")
            _p(f"    response:         {local_counts['response']}")
            _p(f"    total completion: {local_counts['total_completion']}  (server: {su.get('completion_tokens', 'n/a')})")

        results.append({
            "label": label,
            "description": description,
            "prefill_text": prefill_text,
            "prompt": prompt,
            "model": model_name,
            "temperature": temperature,
            "seed": seed,
            "generated_text": raw["generated_text"],
            "full_content_with_prefill": full,
            "thinking_content": thinking_content,
            "response_content": response_content,
            "elapsed_s": raw["elapsed_s"],
            "ttft_s": raw["ttft_s"],
            "stream_chunks": raw["stream_chunks"],
            "usage": raw["usage"],
            "local_token_counts": local_counts,
            "timestamp_utc": raw["timestamp_utc"],
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    endpoints = load_endpoints()
    model_choices = list(endpoints.keys())

    parser = argparse.ArgumentParser(
        description="Test assistant-turn prefills against RunPod vLLM endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(model_choices)}",
    )
    parser.add_argument(
        "prompt", nargs="?",
        help=f"User prompt (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument("--model", "-m", required=True, choices=model_choices)
    parser.add_argument("--system", "-s", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--prefill", "-p", default=None,
        help="Single custom prefill. Omit to run the full default battery.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--t", type=float, default=0.7, dest="temperature")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin.")
    parser.add_argument("--json-out", type=str, default=None, help="Save results to this JSON file.")
    parser.add_argument("--debug", action="store_true", help="Print raw stream chunks.")

    args = parser.parse_args()

    if args.stdin:
        args.prompt = sys.stdin.read().strip()
    if not args.prompt:
        args.prompt = DEFAULT_PROMPT

    args.endpoint_cfg = endpoints[args.model]
    return args


def main():
    args = parse_args()
    cfg = args.endpoint_cfg
    client = build_client(cfg["endpoint_id"])

    prefills = (
        [{"label": "custom", "text": args.prefill,
          "description": f"Custom prefill: {repr(args.prefill)}"}]
        if args.prefill is not None
        else DEFAULT_PREFILLS
    )

    _p(f"\nRunning {len(prefills)} prefill experiment(s) on [{args.model}]")
    _p(f"Prompt: {args.prompt!r}\n")

    results = run_prefill_battery(
        client,
        cfg["model_name"],
        args.prompt,
        prefills,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        debug=args.debug,
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        _p(f"\nResults saved to {out_path}")

    _p(f"\n{'=' * 60}")
    _p(f"  SUMMARY - {args.model}")
    _p(f"{'=' * 60}")
    _p(f"  {'label':<18} {'elapsed':>8}  {'think_tok':>9}  {'resp_tok':>8}  prefill")
    _p(f"  {'-'*18} {'-'*8}  {'-'*9}  {'-'*8}  {'-'*30}")
    for r in results:
        if "error" in r:
            _p(f"  {r['label']:<18} ERROR: {r['error'][:50]}")
            continue
        lc = r.get("local_token_counts") or {}
        elapsed = f"{r['elapsed_s']:.1f}s" if r.get("elapsed_s") else "n/a"
        _p(
            f"  {r['label']:<18} {elapsed:>8}  "
            f"{lc.get('thinking', 'n/a'):>9}  "
            f"{lc.get('response', 'n/a'):>8}  "
            f"{repr(r['prefill_text'])}"
        )
    _p(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
