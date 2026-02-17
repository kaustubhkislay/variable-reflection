"""But-wait continuation loop for eliciting extended reasoning.

After the model's initial response, appends "but wait: " and forces the model
to continue reasoning via assistant prefilling. Repeats until the output token
budget is exhausted.
"""

import time
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class ButWaitResult:
    """Result from a but-wait continuation loop."""
    content: str            # Final continuation's content
    thinking: Optional[str] # Initial response's thinking trace
    full_response: str      # Tagged: [INITIAL]\n...\n[BUT_WAIT_1]\n...
    input_tokens: int       # Sum across all calls
    output_tokens: int      # Sum across all calls
    num_continuations: int  # Number of but-wait iterations


def _build_tagged_response(segments: list) -> str:
    """Build tagged full_response from list of response segments."""
    parts = [f"[INITIAL]\n{segments[0]}"]
    for i, seg in enumerate(segments[1:], 1):
        parts.append(f"[BUT_WAIT_{i}]\n{seg}")
    return "\n\n".join(parts)


def _build_continuation_messages(initial_prompt: str, segments: list) -> list:
    """Build multi-turn messages for continuation with assistant prefill."""
    assistant_content = config.BUT_WAIT_SUFFIX.join(segments) + config.BUT_WAIT_SUFFIX
    return [
        {"role": "user", "content": initial_prompt},
        {"role": "assistant", "content": assistant_content}
    ]


async def but_wait_loop_async(
    initial_prompt: str,
    initial_response,
    api_call_fn,
    token_budget: int,
) -> ButWaitResult:
    """
    Run but-wait continuation loop (async).

    After the initial response, appends BUT_WAIT_SUFFIX and sends as assistant
    prefill to force the model to continue reasoning. Repeats until
    token_budget - total_output < BUT_WAIT_MIN_BUDGET.

    Args:
        initial_prompt: The original user prompt
        initial_response: Response object from initial API call (has .content,
            .thinking, .input_tokens, .output_tokens)
        api_call_fn: Async callable(messages=..., max_tokens=...) -> Response.
            Must accept messages and max_tokens keyword args.
        token_budget: Total output token budget for the experiment
    """
    segments = [initial_response.content]
    total_input = initial_response.input_tokens
    total_output = initial_response.output_tokens
    remaining = token_budget - total_output
    num_continuations = 0

    while remaining >= config.BUT_WAIT_MIN_BUDGET:
        messages = _build_continuation_messages(initial_prompt, segments)
        continuation = await api_call_fn(messages=messages, max_tokens=remaining)
        segments.append(continuation.content)
        total_input += continuation.input_tokens
        total_output += continuation.output_tokens
        remaining = token_budget - total_output
        num_continuations += 1

    return ButWaitResult(
        content=segments[-1],
        thinking=initial_response.thinking,
        full_response=_build_tagged_response(segments),
        input_tokens=total_input,
        output_tokens=total_output,
        num_continuations=num_continuations,
    )


def but_wait_loop_sync(
    initial_prompt: str,
    initial_response,
    api_call_fn,
    token_budget: int,
) -> ButWaitResult:
    """
    Run but-wait continuation loop (sync).

    Same as but_wait_loop_async but for synchronous API calls.

    Args:
        initial_prompt: The original user prompt
        initial_response: Response object from initial API call
        api_call_fn: Sync callable(messages=..., max_tokens=...) -> Response.
            Must accept messages and max_tokens keyword args.
        token_budget: Total output token budget for the experiment
    """
    segments = [initial_response.content]
    total_input = initial_response.input_tokens
    total_output = initial_response.output_tokens
    remaining = token_budget - total_output
    num_continuations = 0

    while remaining >= config.BUT_WAIT_MIN_BUDGET:
        messages = _build_continuation_messages(initial_prompt, segments)
        continuation = api_call_fn(messages=messages, max_tokens=remaining)
        segments.append(continuation.content)
        total_input += continuation.input_tokens
        total_output += continuation.output_tokens
        remaining = token_budget - total_output
        num_continuations += 1

    return ButWaitResult(
        content=segments[-1],
        thinking=initial_response.thinking,
        full_response=_build_tagged_response(segments),
        input_tokens=total_input,
        output_tokens=total_output,
        num_continuations=num_continuations,
    )
