#!/usr/bin/env python3
"""
CLI tool for running experiments against RunPod vLLM serverless endpoints.

Usage:
    python run_experiment.py --model qwen3-32b "What is the capital of France?"
    python run_experiment.py --model qwen3-32b --stream --max-tokens 512 --t 0.3 "Explain quantum computing"
    echo "prompt text" | python run_experiment.py --model qwen3-32b --stdin
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError, InternalServerError, APIStatusError

try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

# All Qwen3 models share the same tokenizer vocabulary.
# Map model-name substrings → HuggingFace tokenizer repo.
# Using the 0.6B repo means only tokenizer files are downloaded (~3 MB), not weights.
_TOKENIZER_ID_MAP = {
    "qwen3": "Qwen/Qwen3-0.6B",
}
_tokenizer_cache: dict = {}


def _get_tokenizer(model_name: str):
    """Return a cached HuggingFace tokenizer for *model_name*, or None."""
    if not _HAS_TRANSFORMERS:
        return None
    tok_id = next(
        (v for k, v in _TOKENIZER_ID_MAP.items() if k in model_name.lower()),
        model_name,
    )
    if tok_id not in _tokenizer_cache:
        print(f"  [tokenizer] Loading '{tok_id}' (first run only)...", flush=True)
        _tokenizer_cache[tok_id] = AutoTokenizer.from_pretrained(tok_id)
    return _tokenizer_cache[tok_id]


def _count_local_tokens(
    messages: list[dict],
    thinking_content: str,
    response_content: str,
    model_name: str,
) -> dict | None:
    """Count tokens locally using the HF tokenizer.

    Uses apply_chat_template for the prompt so special tokens are included,
    matching what the server actually tokenizes.
    """
    tokenizer = _get_tokenizer(model_name)
    if tokenizer is None:
        return None

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    thinking_tokens = len(tokenizer.encode(thinking_content, add_special_tokens=False)) if thinking_content else 0
    response_tokens = len(tokenizer.encode(response_content, add_special_tokens=False)) if response_content else 0

    return {
        "prompt": prompt_tokens,
        "thinking": thinking_tokens,
        "response": response_tokens,
        "total_completion": thinking_tokens + response_tokens,
    }


def _p(*args, **kwargs):
    """Print with flush for Windows compatibility."""
    print(*args, **kwargs, flush=True)


def load_endpoints() -> dict:
    path = SCRIPT_DIR / "endpoints.json"
    if not path.exists():
        print("ERROR: endpoints.json not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_client(endpoint_id: str) -> OpenAI:
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(
        api_key=api_key,
        base_url=f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1",
        timeout=300.0,  # 5 min — RunPod serverless can cold-start
    )


def _parse_thinking(content: str) -> tuple[str, str]:
    """Split content into (thinking_content, response_content).

    Qwen3 wraps chain-of-thought in <think>...</think> at the start of the
    output. Everything after the closing tag is the actual answer.
    Returns empty string for thinking_content when the tag is absent.
    """
    match = re.match(r"<think>(.*?)</think>\s*", content, re.DOTALL)
    if match:
        return match.group(1).strip(), content[match.end():].strip()
    return "", content.strip()


def run_chat(
    client: OpenAI,
    model_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    seed: int | None = None,
    stream: bool = False,
    thinking: bool = True,
    debug: bool = False,
) -> dict:
    """Send a chat completion request and return result with timing info."""
    effective_prompt = prompt if thinking else f"/no_think {prompt}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": effective_prompt})

    params = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=stream,
    )
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if seed is not None:
        params["seed"] = seed
    if stream:
        params["stream_options"] = {"include_usage": True}

    _p(f"\n{'-' * 60}")
    _p(f"  Model:       {model_name}")
    _p(f"  Temperature: {temperature}")
    _p(f"  Max tokens:  {max_tokens or 'default'}")
    _p(f"  Seed:        {seed or 'none'}")
    _p(f"  Stream:      {stream}")
    _p(f"  Thinking:    {'on' if thinking else 'off'}")
    _p(f"{'-' * 60}")
    _p(f"\n[PROMPT]\n{prompt}\n")
    _p(f"{'-' * 60}")
    print("Waiting for response...", flush=True)

    t_start = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    stop_heartbeat = threading.Event()

    def heartbeat():
        while not stop_heartbeat.is_set():
            if stop_heartbeat.wait(10):
                break
            elapsed = time.perf_counter() - t_start
            print(f"  ... still waiting ({elapsed:.0f}s elapsed)", flush=True)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    try:
        if stream:
            result = _run_streaming(client, params, t_start, debug=debug)
        else:
            result = _run_non_streaming(client, params, t_start, debug=debug)
    except APITimeoutError:
        elapsed = time.perf_counter() - t_start
        print(f"\nERROR: Request timed out after {elapsed:.0f}s.", file=sys.stderr)
        print("The endpoint may be in cold start. Try again in a minute.", file=sys.stderr)
        sys.exit(1)
    except APIConnectionError as e:
        print(f"\nERROR: Connection failed: {e}", file=sys.stderr)
        sys.exit(1)
    except InternalServerError as e:
        elapsed = time.perf_counter() - t_start
        print(f"\nERROR: Server returned 500 after {elapsed:.0f}s.", file=sys.stderr)
        print(f"Detail: {e}", file=sys.stderr)
        print("This often means the worker is still cold-starting or ran out of resources.", file=sys.stderr)
        print("Try: python run_experiment.py --ping -m <model>  to check endpoint status.", file=sys.stderr)
        sys.exit(1)
    except APIStatusError as e:
        elapsed = time.perf_counter() - t_start
        print(f"\nERROR: API returned status {e.status_code} after {elapsed:.0f}s.", file=sys.stderr)
        print(f"Detail: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        stop_heartbeat.set()
        hb.join(timeout=1)

    raw_content = result.get("content") or result.get("full_content", "")
    thinking_content, response_content = _parse_thinking(raw_content)

    local_counts = _count_local_tokens(messages, thinking_content, response_content, model_name)
    if local_counts:
        server_usage = result.get("usage") or {}
        _p(f"\n  [local tokenizer counts]")
        _p(f"    prompt:           {local_counts['prompt']}  (server: {server_usage.get('prompt_tokens', 'n/a')})")
        _p(f"    thinking:         {local_counts['thinking']}")
        _p(f"    response:         {local_counts['response']}")
        _p(f"    total completion: {local_counts['total_completion']}  (server: {server_usage.get('completion_tokens', 'n/a')})")

    result.update({
        "prompt": prompt,
        "model": model_name,
        "timestamp_utc": timestamp,
        "temperature": temperature,
        "seed": seed,
        "thinking_enabled": thinking,
        "thinking_content": thinking_content,
        "response_content": response_content,
        "local_token_counts": local_counts,
    })
    return result


def _run_streaming(client: OpenAI, params: dict, t_start: float, *, debug: bool = False) -> dict:
    response_stream = client.chat.completions.create(**params)

    collected_content = []
    first_token_time = None
    token_count = 0
    usage = None

    for i, chunk in enumerate(response_stream):
        now = time.perf_counter()

        if debug and i < 5:
            _p(f"  [DEBUG chunk {i}] {chunk}")

        # Final chunk from stream_options={"include_usage": True} has no choices
        if not chunk.choices:
            if chunk.usage:
                usage = chunk.usage
            continue

        delta = chunk.choices[0].delta
        if delta.content:
            if first_token_time is None:
                first_token_time = now
                print(f"[TTFT: {first_token_time - t_start:.2f}s] ", end="", flush=True)
            print(delta.content, end="", flush=True)
            collected_content.append(delta.content)
            token_count += 1

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    full_text = "".join(collected_content)

    if debug and token_count == 0:
        _p("  [DEBUG] No content tokens received.")

    _p(f"\n\n{'-' * 60}")
    _p(f"  Total time:          {elapsed:.2f}s")
    if first_token_time is not None:
        _p(f"  Time to first token: {first_token_time - t_start:.2f}s")
        gen_time = t_end - first_token_time
        if token_count > 1:
            _p(f"  Generation time:     {gen_time:.2f}s (~{token_count} chunks)")
            _p(f"  Throughput:          ~{token_count / gen_time:.1f} chunks/s")
    if usage:
        _p(f"  Prompt tokens:     {usage.prompt_tokens}")
        _p(f"  Completion tokens: {usage.completion_tokens}")
        _p(f"  Total tokens:      {usage.total_tokens}")
    _p(f"{'-' * 60}\n")

    return {
        "full_content": full_text,
        "elapsed_s": elapsed,
        "ttft_s": (first_token_time - t_start) if first_token_time else None,
        "stream_chunks": token_count,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        } if usage else None,
    }


def _run_non_streaming(client: OpenAI, params: dict, t_start: float, *, debug: bool = False) -> dict:
    response = client.chat.completions.create(**params)
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    if debug:
        _p(f"  [DEBUG response] {response}")

    content = response.choices[0].message.content or ""
    usage = response.usage

    _p(content)

    _p(f"\n{'-' * 60}")
    _p(f"  Total time:        {elapsed:.2f}s")
    if usage:
        _p(f"  Prompt tokens:     {usage.prompt_tokens}")
        _p(f"  Completion tokens: {usage.completion_tokens}")
        _p(f"  Total tokens:      {usage.total_tokens}")
        if usage.completion_tokens > 0:
            _p(f"  Tokens/sec:        ~{usage.completion_tokens / elapsed:.1f}")
    _p(f"{'-' * 60}\n")

    return {
        "full_content": content,
        "elapsed_s": elapsed,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        } if usage else None,
    }


def parse_args() -> argparse.Namespace:
    endpoints = load_endpoints()
    model_choices = list(endpoints.keys())

    parser = argparse.ArgumentParser(
        description="Run experiments against RunPod vLLM endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(model_choices)}",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send. Omit if using --stdin.",
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=model_choices,
        help="Model alias from endpoints.json.",
    )
    parser.add_argument(
        "--system", "-s",
        default=None,
        help="Optional system prompt.",
    )

    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Enable streaming (default: off).",
    )
    stream_group.add_argument(
        "--no-stream",
        action="store_true",
        help="Explicitly disable streaming.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens to generate (default: model default).",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.7,
        dest="temperature",
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read prompt from stdin.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write result JSON to this file path.",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Just check if the endpoint is alive (list models) and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw API response/chunk data for troubleshooting.",
    )
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument(
        "--think",
        action="store_true",
        default=True,
        dest="thinking",
        help="Enable thinking / chain-of-thought (default: on).",
    )
    think_group.add_argument(
        "--no-think",
        action="store_false",
        dest="thinking",
        help="Disable thinking (prepends /no_think to the prompt for Qwen3 models).",
    )

    args = parser.parse_args()

    if args.ping:
        args.prompt = None
    elif args.stdin:
        args.prompt = sys.stdin.read().strip()

    if not args.ping and not args.prompt:
        parser.error("Provide a prompt as a positional arg or use --stdin.")

    args.endpoint_cfg = endpoints[args.model]
    return args


def ping_endpoint(client: OpenAI, alias: str, expected_model: str | None = None):
    """Check endpoint health and validate model name."""
    _p(f"\nPinging endpoint for '{alias}'...")
    t_start = time.perf_counter()
    try:
        models = client.models.list()
        elapsed = time.perf_counter() - t_start
        model_ids = [m.id for m in models]
        _p(f"  Endpoint is UP ({elapsed:.2f}s)")
        _p(f"  Available models: {model_ids}")

        if expected_model and expected_model not in model_ids:
            _p(f"\n  WARNING: endpoints.json has model_name '{expected_model}'")
            _p(f"           but endpoint serves: {model_ids}")
            if model_ids:
                _p(f"           Update endpoints.json to use '{model_ids[0]}'")
            return False

        return True
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        _p(f"  Endpoint UNREACHABLE after {elapsed:.0f}s: {e}")
        return False


def main():
    args = parse_args()
    cfg = args.endpoint_cfg

    client = build_client(cfg["endpoint_id"])

    if args.ping:
        ok = ping_endpoint(client, args.model, expected_model=cfg["model_name"])
        sys.exit(0 if ok else 1)

    result = run_chat(
        client,
        cfg["model_name"],
        args.prompt,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        stream=args.stream,
        thinking=args.thinking,
        debug=args.debug,
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {out_path}")


if __name__ == "__main__":
    main()
