"""API wrapper for Claude with thinking toggle."""

import anthropic
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

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

def call_claude(
    prompt: str,
    thinking_enabled: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0
) -> APIResponse:
    """
    Call Claude API with optional extended thinking.

    Args:
        prompt: The user prompt
        thinking_enabled: Whether to enable extended thinking
        max_retries: Number of retries on failure
        retry_delay: Seconds to wait between retries

    Returns:
        APIResponse with content, thinking, and token counts
    """

    kwargs = {
        "model": config.MODEL,
        "max_tokens": (config.MAX_TOKENS_WITH_THINKING if thinking_enabled
                       else config.MAX_TOKENS_NO_THINKING),
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


def call_with_rate_limit(prompt: str, thinking_enabled: bool = False) -> APIResponse:
    """Call API with rate limiting."""
    time.sleep(60 / config.CALLS_PER_MINUTE)  # Simple rate limiting
    return call_claude(prompt, thinking_enabled)
