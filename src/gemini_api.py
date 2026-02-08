"""API wrapper for Gemini 3 Flash via OpenRouter with thinking level control."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Any

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None

import config


@dataclass
class GeminiResponse:
    """Standardized response from Gemini API via OpenRouter."""
    content: str
    thinking: Optional[str]
    input_tokens: int
    output_tokens: int


# Valid Gemini thinking levels
VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


def validate_thinking_level(thinking_level: str) -> str:
    """Validate and return thinking_level, defaulting to 'low' if invalid."""
    if thinking_level in VALID_THINKING_LEVELS:
        return thinking_level
    return "low"


class GeminiAsyncRateLimiter:
    """Thread-safe async rate limiter for Gemini API."""

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


# Initialize clients and rate limiter
_sync_client = None
_async_client = None
_rate_limiter = None


def _get_sync_client():
    """Lazy initialization of OpenRouter sync client."""
    global _sync_client
    if _sync_client is None:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _sync_client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL
        )
    return _sync_client


def _get_async_client():
    """Lazy initialization of OpenRouter async client."""
    global _async_client
    if _async_client is None:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _async_client = AsyncOpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL
        )
    return _async_client


def _get_rate_limiter():
    """Get or create rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = GeminiAsyncRateLimiter(config.CALLS_PER_MINUTE)
    return _rate_limiter


def _build_messages_and_params(prompt: str, thinking_level: str, max_tokens: int) -> tuple:
    """Build messages and extra parameters for OpenRouter API call."""
    messages = [{"role": "user", "content": prompt}]

    # OpenRouter passes provider-specific params via extra_body
    # reasoning.effort maps to Gemini's thinkingLevel parameter
    extra_body = {
        "provider": {
            "order": ["Google"],  # Prefer Google's Gemini
            "allow_fallbacks": False
        },
        "reasoning": {
            "effort": validate_thinking_level(thinking_level),
            "exclude": False  # Return thinking traces in response
        }
    }

    return messages, extra_body


def _parse_openrouter_response(response: Any) -> GeminiResponse:
    """Parse OpenRouter API response into standardized format."""
    content = ""
    thinking = None

    # Extract content and reasoning trace from response
    if response.choices and len(response.choices) > 0:
        message = response.choices[0].message
        if message.content:
            content = message.content
        # OpenRouter returns thinking traces in message.reasoning
        thinking = getattr(message, 'reasoning', None)

    # Get token counts
    input_tokens = 0
    output_tokens = 0
    if response.usage:
        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0

    return GeminiResponse(
        content=content,
        thinking=thinking,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )


def call_gemini(
    prompt: str,
    thinking_level: str = "low",
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens: int = None
) -> GeminiResponse:
    """
    Call Gemini API via OpenRouter with thinking level control (synchronous).

    Args:
        prompt: The user prompt
        thinking_level: Thinking level (minimal, low, medium, high)
        max_retries: Number of retries on failure
        retry_delay: Seconds to wait between retries
        max_tokens: Max output tokens (default from config)

    Returns:
        GeminiResponse with content, thinking, and token counts
    """
    client = _get_sync_client()
    max_tokens = max_tokens or config.GEMINI_MAX_TOKENS
    messages, extra_body = _build_messages_and_params(prompt, thinking_level, max_tokens)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.GEMINI_MODEL,
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


async def call_gemini_async(
    prompt: str,
    thinking_level: str = "low",
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens: int = None
) -> GeminiResponse:
    """
    Call Gemini API via OpenRouter with thinking level control (asynchronous).

    Args:
        prompt: The user prompt
        thinking_level: Thinking level (minimal, low, medium, high)
        max_retries: Number of retries on failure
        retry_delay: Seconds to wait between retries
        max_tokens: Max output tokens (default from config)

    Returns:
        GeminiResponse with content, thinking, and token counts
    """
    client = _get_async_client()
    max_tokens = max_tokens or config.GEMINI_MAX_TOKENS
    messages, extra_body = _build_messages_and_params(prompt, thinking_level, max_tokens)

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=config.GEMINI_MODEL,
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


def call_gemini_with_rate_limit(
    prompt: str,
    thinking_level: str = "low",
    max_tokens: int = None
) -> GeminiResponse:
    """Call Gemini API via OpenRouter with rate limiting (synchronous)."""
    time.sleep(60 / config.CALLS_PER_MINUTE)
    return call_gemini(prompt, thinking_level, max_tokens=max_tokens)


async def call_gemini_with_rate_limit_async(
    prompt: str,
    thinking_level: str = "low",
    max_tokens: int = None
) -> GeminiResponse:
    """Call Gemini API via OpenRouter with rate limiting (asynchronous)."""
    rate_limiter = _get_rate_limiter()
    await rate_limiter.acquire()
    return await call_gemini_async(prompt, thinking_level, max_tokens=max_tokens)


def reset_gemini_rate_limiter():
    """Reset the global rate limiter stats."""
    global _rate_limiter
    _rate_limiter = GeminiAsyncRateLimiter(config.CALLS_PER_MINUTE)


def get_gemini_rate_stats():
    """Get current rate limiter statistics."""
    rate_limiter = _get_rate_limiter()
    return rate_limiter.get_stats()


def is_gemini_available() -> bool:
    """Check if Gemini API via OpenRouter is available and configured."""
    return OPENAI_AVAILABLE and bool(config.OPENROUTER_API_KEY)
