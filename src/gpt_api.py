"""API wrapper for GPT 5.2 via OpenRouter with reasoning level control.

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
class GPTResponse:
    """Standardized response from GPT API via OpenRouter."""
    content: str
    thinking: Optional[str]
    input_tokens: int
    output_tokens: int


# Valid GPT reasoning levels
VALID_THINKING_LEVELS = {"none", "low", "medium", "high", "xhigh"}


def validate_thinking_level(thinking_level: str) -> str:
    """Validate and return thinking_level, defaulting to 'low' if invalid."""
    if thinking_level in VALID_THINKING_LEVELS:
        return thinking_level
    return "low"


def _build_messages_and_params(prompt: str, thinking_level: str, max_tokens: int) -> tuple:
    """Build messages and extra parameters for OpenRouter API call."""
    messages = [{"role": "user", "content": prompt}]

    extra_body = {
        "provider": {
            "order": ["OpenAI"],
            "allow_fallbacks": False
        },
        "reasoning": {
            "effort": validate_thinking_level(thinking_level),
            "exclude": False  # Return thinking traces in response
        }
    }

    return messages, extra_body


def _parse_openrouter_response(response: Any) -> GPTResponse:
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

    return GPTResponse(
        content=content,
        thinking=thinking,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )


def call_gpt(
    prompt: str,
    thinking_level: str = "low",
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens: int = None
) -> GPTResponse:
    """Call GPT 5.2 via OpenRouter with reasoning level control (synchronous)."""
    client = _get_sync_client()
    max_tokens = max_tokens or config.GPT_MAX_TOKENS
    messages, extra_body = _build_messages_and_params(prompt, thinking_level, max_tokens)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.GPT_MODEL,
                messages=messages,
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


async def call_gpt_async(
    prompt: str,
    thinking_level: str = "low",
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens: int = None
) -> GPTResponse:
    """Call GPT 5.2 via OpenRouter with reasoning level control (asynchronous)."""
    client = _get_async_client()
    max_tokens = max_tokens or config.GPT_MAX_TOKENS
    messages, extra_body = _build_messages_and_params(prompt, thinking_level, max_tokens)

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=config.GPT_MODEL,
                messages=messages,
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


def call_gpt_with_rate_limit(
    prompt: str,
    thinking_level: str = "low",
    max_tokens: int = None
) -> GPTResponse:
    """Call GPT 5.2 via OpenRouter with rate limiting (synchronous)."""
    time.sleep(60 / config.CALLS_PER_MINUTE)
    return call_gpt(prompt, thinking_level, max_tokens=max_tokens)


async def call_gpt_with_rate_limit_async(
    prompt: str,
    thinking_level: str = "low",
    max_tokens: int = None
) -> GPTResponse:
    """Call GPT 5.2 via OpenRouter with rate limiting (asynchronous)."""
    rate_limiter = _get_rate_limiter()
    await rate_limiter.acquire()
    return await call_gpt_async(prompt, thinking_level, max_tokens=max_tokens)


def is_gpt_available() -> bool:
    """Check if GPT via OpenRouter is available and configured."""
    return OPENAI_AVAILABLE and bool(config.OPENROUTER_API_KEY)
