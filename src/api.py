"""API wrapper for Claude with thinking toggle and async support."""

import anthropic
import asyncio
import time
from dataclasses import dataclass
from typing import Optional
import config

@dataclass
class APIResponse:
    content: str
    thinking: Optional[str]
    input_tokens: int
    output_tokens: int


# Synchronous client (for backwards compatibility)
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Async client
async_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)


class AsyncRateLimiter:
    """Thread-safe async rate limiter."""

    def __init__(self, calls_per_minute: int):
        self.delay = 60 / calls_per_minute
        self.last_call = 0
        self.lock = asyncio.Lock()
        self.call_count = 0
        self.start_time = None

    async def acquire(self):
        async with self.lock:
            now = time.time()
            if self.start_time is None:
                self.start_time = now

            wait_time = self.last_call + self.delay - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self.last_call = time.time()
            self.call_count += 1

    def get_stats(self):
        if self.start_time is None:
            return {"calls": 0, "elapsed": 0, "rate": 0}
        elapsed = time.time() - self.start_time
        rate = self.call_count / (elapsed / 60) if elapsed > 0 else 0
        return {
            "calls": self.call_count,
            "elapsed": elapsed,
            "rate": round(rate, 1)
        }


# Global async rate limiter
rate_limiter = AsyncRateLimiter(config.CALLS_PER_MINUTE)


def call_claude(
    prompt: str,
    thinking_enabled: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens_override: int = None
) -> APIResponse:
    """
    Call Claude API with optional extended thinking (synchronous).

    Args:
        prompt: The user prompt
        thinking_enabled: Whether to enable extended thinking
        max_retries: Number of retries on failure
        retry_delay: Seconds to wait between retries
        max_tokens_override: Optional override for max_tokens (useful for Level 0)

    Returns:
        APIResponse with content, thinking, and token counts
    """

    # Determine max_tokens
    if max_tokens_override is not None:
        max_tokens = max_tokens_override
    elif thinking_enabled:
        max_tokens = config.MAX_TOKENS_WITH_THINKING
    else:
        max_tokens = config.MAX_TOKENS_NO_THINKING

    kwargs = {
        "model": config.MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    # Add thinking configuration if enabled
    if thinking_enabled:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.THINKING_BUDGET
        }

    # Add temperature if not using thinking (thinking requires temperature=1)
    if not thinking_enabled:
        kwargs["temperature"] = config.TEMPERATURE

    # Retry loop
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)

            # Parse response blocks
            content = ""
            thinking = None

            for block in response.content:
                if block.type == "thinking":
                    thinking = block.thinking
                elif block.type == "text":
                    content = block.text

            return APIResponse(
                content=content,
                thinking=thinking,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                print(f"Rate limited, waiting {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise

        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                print(f"API error: {e}, retrying...")
                time.sleep(retry_delay)
            else:
                raise

    raise RuntimeError("Max retries exceeded")


async def call_claude_async(
    prompt: str,
    thinking_enabled: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens_override: int = None
) -> APIResponse:
    """
    Call Claude API with optional extended thinking (asynchronous).

    Args:
        prompt: The user prompt
        thinking_enabled: Whether to enable extended thinking
        max_retries: Number of retries on failure
        retry_delay: Seconds to wait between retries
        max_tokens_override: Optional override for max_tokens (useful for Level 0)

    Returns:
        APIResponse with content, thinking, and token counts
    """

    # Determine max_tokens
    if max_tokens_override is not None:
        max_tokens = max_tokens_override
    elif thinking_enabled:
        max_tokens = config.MAX_TOKENS_WITH_THINKING
    else:
        max_tokens = config.MAX_TOKENS_NO_THINKING

    kwargs = {
        "model": config.MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    # Add thinking configuration if enabled
    if thinking_enabled:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.THINKING_BUDGET
        }

    # Add temperature if not using thinking
    if not thinking_enabled:
        kwargs["temperature"] = config.TEMPERATURE

    # Retry loop
    for attempt in range(max_retries):
        try:
            response = await async_client.messages.create(**kwargs)

            # Parse response blocks
            content = ""
            thinking = None

            for block in response.content:
                if block.type == "thinking":
                    thinking = block.thinking
                elif block.type == "text":
                    content = block.text

            return APIResponse(
                content=content,
                thinking=thinking,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                print(f"Rate limited, waiting {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise

        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                print(f"API error: {e}, retrying...")
                await asyncio.sleep(retry_delay)
            else:
                raise

    raise RuntimeError("Max retries exceeded")


def call_with_rate_limit(prompt: str, thinking_enabled: bool = False, max_tokens_override: int = None) -> APIResponse:
    """Call API with rate limiting (synchronous)."""
    time.sleep(60 / config.CALLS_PER_MINUTE)  # Simple rate limiting
    return call_claude(prompt, thinking_enabled, max_tokens_override=max_tokens_override)


async def call_with_rate_limit_async(prompt: str, thinking_enabled: bool = False, max_tokens_override: int = None) -> APIResponse:
    """Call API with rate limiting (asynchronous)."""
    await rate_limiter.acquire()
    return await call_claude_async(prompt, thinking_enabled, max_tokens_override=max_tokens_override)


def reset_rate_limiter():
    """Reset the global rate limiter stats."""
    global rate_limiter
    rate_limiter = AsyncRateLimiter(config.CALLS_PER_MINUTE)


def get_rate_stats():
    """Get current rate limiter statistics."""
    return rate_limiter.get_stats()
