"""LLM-as-Judge for detecting flip-flop in reasoning traces.

Uses Gemini 3 Flash to parse reasoning traces and identify position changes
(flip-flopping) during moral reasoning.
"""

import re
from typing import Dict, Optional

import pandas as pd

import config
from prompts import JUDGE_FLIP_FLOP_PROMPT
from src.gemini_api import call_gemini_with_rate_limit_async


def _build_scenario_description(row: dict, benchmark: str) -> str:
    """Build the scenario description for the judge prompt."""
    if benchmark == "ethics":
        return f'Scenario: {row["scenario"]}\nQuestion: Is this wrong or not wrong?'
    elif benchmark == "moralchoice":
        return (
            f'Context: {row["context"]}\n'
            f'A) {row["option_a"]}\n'
            f'B) {row["option_b"]}\n'
            f'Question: Which is more morally acceptable?'
        )
    elif benchmark == "morables":
        return (
            f'Fable: {row["fable"]}\n'
            f'Question: What is the moral of this story?\n'
            f'Options: A through E (multiple choice)'
        )
    return str(row.get("scenario", row.get("context", row.get("fable", ""))))


def _build_answer_space(benchmark: str) -> str:
    """Build the answer space description."""
    if benchmark == "ethics":
        return '"wrong" or "not wrong"'
    elif benchmark == "moralchoice":
        return '"A" or "B"'
    elif benchmark == "morables":
        return '"A", "B", "C", "D", or "E"'
    return "unknown"


def _build_flip_type_instruction(benchmark: str) -> str:
    """Build flip_flop_type instruction based on benchmark.

    MoralChoice has no ground truth, so flip type is always neutral.
    """
    if benchmark == "moralchoice":
        return 'neutral (this benchmark has no ground truth correct answer)'
    return 'none, beneficial, harmful, or neutral'


def _assemble_reasoning_trace(row: dict) -> str:
    """Assemble the full reasoning trace from a results row.

    Combines thinking_content and response when both exist.
    """
    parts = []

    thinking = row.get('thinking_content')
    if pd.notna(thinking) and str(thinking).strip():
        parts.append(f"[INTERNAL THINKING]\n{thinking}")

    response = row.get('response')
    if pd.notna(response) and str(response).strip():
        parts.append(f"[VISIBLE RESPONSE]\n{response}")

    return "\n\n".join(parts)


def should_judge_row(row: dict) -> bool:
    """Determine if a row has enough reasoning content to judge.

    Requires at least JUDGE_MIN_TRACE_WORDS words across
    response and thinking_content combined.
    """
    if pd.isna(row.get('extracted_answer')):
        return False

    response = str(row.get('response', ''))
    thinking = str(row.get('thinking_content', ''))

    response_words = len(response.split()) if response != 'nan' else 0
    thinking_words = len(thinking.split()) if thinking != 'nan' else 0

    return (response_words + thinking_words) >= config.JUDGE_MIN_TRACE_WORDS


def build_judge_prompt(row: dict, benchmark: str) -> str:
    """Build the full judge prompt for a single row."""
    scenario_description = _build_scenario_description(row, benchmark)
    answer_space = _build_answer_space(benchmark)
    final_answer = str(row.get('extracted_answer', 'unknown'))
    reasoning_trace = _assemble_reasoning_trace(row)
    flip_type_instruction = _build_flip_type_instruction(benchmark)

    return JUDGE_FLIP_FLOP_PROMPT.format(
        scenario_description=scenario_description,
        answer_space=answer_space,
        final_answer=final_answer,
        reasoning_trace=reasoning_trace,
        flip_type_instruction=flip_type_instruction,
    )


def parse_judge_response(response_text: str) -> dict:
    """Parse structured judge response into a dict.

    Extracts XML-tagged fields from the judge's response,
    following the same regex pattern as src/extraction.py.
    """
    result = {
        'flip_flop_detected': None,
        'num_position_changes': None,
        'initial_lean': None,
        'final_lean': None,
        'trajectory': None,
        'flip_flop_type': None,
        'judge_summary': None,
        'judge_parse_error': False,
    }

    if response_text is None:
        result['judge_parse_error'] = True
        return result

    text = response_text.strip()

    tag_patterns = {
        'flip_flop_detected': r'<flip_flop_detected>\s*(yes|no)\s*</flip_flop_detected>',
        'num_position_changes': r'<num_position_changes>\s*(\d+)\s*</num_position_changes>',
        'initial_lean': r'<initial_lean>\s*(.+?)\s*</initial_lean>',
        'final_lean': r'<final_lean>\s*(.+?)\s*</final_lean>',
        'trajectory': r'<trajectory>\s*(.+?)\s*</trajectory>',
        'flip_flop_type': r'<flip_flop_type>\s*(none|beneficial|harmful|neutral)\s*</flip_flop_type>',
        'judge_summary': r'<summary>\s*(.+?)\s*</summary>',
    }

    for field, pattern in tag_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            if field == 'flip_flop_detected':
                value = value.lower() == 'yes'
            elif field == 'num_position_changes':
                value = int(value)
            result[field] = value
        else:
            result['judge_parse_error'] = True

    return result


async def judge_single_row_async(row: dict, benchmark: str) -> dict:
    """Judge a single row for flip-flopping using Gemini 3 Flash.

    Returns a dict with judge verdict fields and token usage.
    """
    prompt = build_judge_prompt(row, benchmark)

    response = await call_gemini_with_rate_limit_async(
        prompt,
        thinking_level=config.JUDGE_THINKING_LEVEL,
        max_tokens=config.JUDGE_MAX_TOKENS,
    )

    result = parse_judge_response(response.content)
    result['judge_input_tokens'] = response.input_tokens
    result['judge_output_tokens'] = response.output_tokens
    return result
