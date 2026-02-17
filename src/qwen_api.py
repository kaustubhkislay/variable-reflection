"""API wrapper for Qwen 3 via OpenRouter with reasoning token budget control.

Unlike GPT/Gemini which use named reasoning effort levels, Qwen uses a numeric
reasoning token budget (e.g. 1024, 2048, 4096, 8192, 16384, 32768).

Reuses OpenRouter clients and rate limiter from gemini_api.py.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Any

import config
from src.gemini_api import (
    _get_sync_client,
    _get_async_client,
    _get_rate_limiter,
    OPENAI_AVAILABLE,
)


@dataclass
class QwenResponse:
    """Standardized response from Qwen API via OpenRouter."""
    content: str
    thinking: Optional[str]
    input_tokens: int
    output_tokens: int


# Valid Qwen reasoning token budgets
VALID_THINKING_LEVELS = {1024, 2048, 4096, 8192, 16384, 32768}


def validate_thinking_level(thinking_level: int) -> int:
    """Validate and return thinking_level, defaulting to 1024 if invalid."""
    if isinstance(thinking_level, int) and thinking_level in VALID_THINKING_LEVELS:
        return thinking_level
    return 1024


def _build_messages_and_params(prompt: str, thinking_level: int, max_tokens: int) -> tuple:
    """Build messages and extra parameters for OpenRouter API call.

    Unlike GPT/Gemini which use reasoning.effort (named levels),
    Qwen uses reasoning.max_tokens (numeric token budget).
    """
    messages = [{"role": "user", "content": prompt}]

    extra_body = {
        "reasoning": {
            "max_tokens": validate_thinking_level(thinking_level),
            "exclude": False  # Return thinking traces in response
        }
    }

    return messages, extra_body


def _parse_openrouter_response(response: Any) -> QwenResponse:
    """Parse OpenRouter API response into standardized format."""
    content = ""
    thinking = None

    if response.choices and len(response.choices) > 0:
        message = response.choices[0].message
        if message.content:
            content = message.content
        thinking = getattr(message, 'reasoning', None)

    input_tokens = 0
    output_tokens = 0
    if response.usage:
        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0

    return QwenResponse(
        content=content,
        thinking=thinking,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )


def call_qwen(
    prompt: str,
    thinking_level: int = 1024,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens: int = None,
    model: str = None,
    messages: list = None
) -> QwenResponse:
    """Call Qwen 3 via OpenRouter with reasoning token budget (synchronous).

    Args:
        prompt: The user prompt
        thinking_level: Reasoning token budget (1024, 2048, 4096, 8192, 16384, 32768)
        max_retries: Number of retries on failure
        retry_delay: Seconds to wait between retries
        max_tokens: Max output tokens (default from config)
        model: OpenRouter model ID (required — one of QWEN_*_MODEL constants)

    Returns:
        QwenResponse with content, thinking, and token counts
    """
    client = _get_sync_client()
    max_tokens = max_tokens or config.QWEN_MAX_TOKENS
    if model is None:
        raise ValueError("model parameter is required for Qwen (specify QWEN_4B_MODEL, etc.)")
    if messages is not None:
        api_messages = messages
    else:
        api_messages, _ = _build_messages_and_params(prompt, thinking_level, max_tokens)
    _, extra_body = _build_messages_and_params(prompt, thinking_level, max_tokens)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=max_tokens,
                extra_body=extra_body
            )
            return _parse_openrouter_response(response)

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    print(f"OpenRouter rate limited, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
            elif attempt < max_retries - 1:
                print(f"OpenRouter API error: {e}, retrying...")
                time.sleep(retry_delay)
            else:
                raise

    raise RuntimeError("Max retries exceeded for OpenRouter API")


async def call_qwen_async(
    prompt: str,
    thinking_level: int = 1024,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens: int = None,
    model: str = None,
    messages: list = None
) -> QwenResponse:
    """Call Qwen 3 via OpenRouter with reasoning token budget (asynchronous).

    Args:
        prompt: The user prompt
        thinking_level: Reasoning token budget (1024, 2048, 4096, 8192, 16384, 32768)
        max_retries: Number of retries on failure
        retry_delay: Seconds to wait between retries
        max_tokens: Max output tokens (default from config)
        model: OpenRouter model ID (required)

    Returns:
        QwenResponse with content, thinking, and token counts
    """
    client = _get_async_client()
    max_tokens = max_tokens or config.QWEN_MAX_TOKENS
    if model is None:
        raise ValueError("model parameter is required for Qwen")
    if messages is not None:
        api_messages = messages
    else:
        api_messages, _ = _build_messages_and_params(prompt, thinking_level, max_tokens)
    _, extra_body = _build_messages_and_params(prompt, thinking_level, max_tokens)

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=max_tokens,
                extra_body=extra_body
            )
            return _parse_openrouter_response(response)

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    print(f"OpenRouter rate limited, waiting {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
            elif attempt < max_retries - 1:
                print(f"OpenRouter API error: {e}, retrying...")
                await asyncio.sleep(retry_delay)
            else:
                raise

    raise RuntimeError("Max retries exceeded for OpenRouter API")


def call_qwen_with_rate_limit(
    prompt: str,
    thinking_level: int = 1024,
    max_tokens: int = None,
    model: str = None,
    messages: list = None
) -> QwenResponse:
    """Call Qwen via OpenRouter with rate limiting (synchronous)."""
    time.sleep(60 / config.CALLS_PER_MINUTE)
    return call_qwen(prompt, thinking_level, max_tokens=max_tokens, model=model, messages=messages)


async def call_qwen_with_rate_limit_async(
    prompt: str,
    thinking_level: int = 1024,
    max_tokens: int = None,
    model: str = None,
    messages: list = None
) -> QwenResponse:
    """Call Qwen via OpenRouter with rate limiting (asynchronous)."""
    rate_limiter = _get_rate_limiter()
    await rate_limiter.acquire()
    return await call_qwen_async(prompt, thinking_level, max_tokens=max_tokens, model=model, messages=messages)


def is_qwen_available() -> bool:
    """Check if Qwen via OpenRouter is available and configured."""
    return OPENAI_AVAILABLE and bool(config.OPENROUTER_API_KEY)
