"""API wrapper for Claude via OpenRouter with thinking toggle."""

import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
import config

@dataclass
class APIResponse:
    content: str
    thinking: Optional[str]
    input_tokens: int
    output_tokens: int

client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
)

def call_claude(
    prompt: str,
    thinking_enabled: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0
) -> APIResponse:
    """
    Call Claude API via OpenRouter with optional extended thinking.

    Args:
        prompt: The user prompt
        thinking_enabled: Whether to enable extended thinking
        max_retries: Number of retries on failure
        retry_delay: Seconds to wait between retries

    Returns:
        APIResponse with content, thinking, and token counts
    """

    max_tokens = (config.MAX_TOKENS_WITH_THINKING if thinking_enabled
                  else config.MAX_TOKENS_NO_THINKING)

    # Build request parameters
    kwargs = {
        "model": config.MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.TEMPERATURE,
    }

    # Add extended thinking via extra_body if enabled
    # OpenRouter passes provider-specific params through extra_body
    if thinking_enabled:
        kwargs["extra_body"] = {
            "anthropic": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": config.THINKING_BUDGET
                }
            }
        }
        # Temperature must be 1 for extended thinking
        kwargs["temperature"] = 1

    # Retry loop
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)

            # Parse response
            content = ""
            thinking = None

            # Extract content from response
            if response.choices and response.choices[0].message:
                message = response.choices[0].message
                content = message.content or ""

                # Check for thinking in the response (if available)
                # OpenRouter may return thinking in different formats
                if hasattr(message, 'thinking'):
                    thinking = message.thinking

            # Get token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return APIResponse(
                content=content,
                thinking=thinking,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str:
                if attempt < max_retries - 1:
                    print(f"Rate limited, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
            else:
                if attempt < max_retries - 1:
                    print(f"API error: {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    raise

    raise RuntimeError("Max retries exceeded")


def call_with_rate_limit(prompt: str, thinking_enabled: bool = False) -> APIResponse:
    """Call API with rate limiting."""
    time.sleep(60 / config.CALLS_PER_MINUTE)  # Simple rate limiting
    return call_claude(prompt, thinking_enabled)
