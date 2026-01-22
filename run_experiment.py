"""Run full experiment."""

import pandas as pd
import os
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time

# Load environment variables from .env
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            match = re.match(r'(\w+)\s*=\s*["\']?([^"\']+)["\']?', line)
            if match:
                os.environ[match.group(1)] = match.group(2)

from prompts import get_ethics_prompt, get_moralchoice_prompt
from src.api import call_with_rate_limit, APIResponse
from src.extraction import (
    extract_ethics_answer,
    extract_moralchoice_with_confidence,
    count_reasoning_markers,
    count_uncertainty_markers
)
import config

# Full experiment settings
LEVELS = [0, 1, 2, 3, 4, 5]
THINKING_CONDITIONS = [False, True]
N_RUNS = config.N_RUNS

# Sample size per benchmark (set to None for full dataset)
SAMPLE_SIZE = 100


def run_single_item_ethics(row, level, thinking):
    """Run single ETHICS item at given condition."""

    if level == 5:
        # Two-pass
        prompt1 = get_ethics_prompt(5, row['scenario'])
        response1 = call_with_rate_limit(prompt1, thinking)

        prompt2 = get_ethics_prompt(5, row['scenario'], response1.content)
        response2 = call_with_rate_limit(prompt2, thinking)

        return {
            'content': response2.content,
            'thinking': response2.thinking,
            'full_response': f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}",
            'input_tokens': response1.input_tokens + response2.input_tokens,
            'output_tokens': response1.output_tokens + response2.output_tokens,
        }
    else:
        prompt = get_ethics_prompt(level, row['scenario'])
        response = call_with_rate_limit(prompt, thinking)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


def run_single_item_moralchoice(row, level, thinking, include_confidence=True):
    """Run single MoralChoice item at given condition.

    Args:
        row: DataFrame row with context, option_a, option_b
        level: Reflection level (0-5)
        thinking: Whether to enable extended thinking
        include_confidence: If True, prompt asks for both answer and confidence

    Returns:
        Dict with content, thinking, full_response, input_tokens, output_tokens
    """
    context = row.get('context', '')

    if level == 5:
        prompt1 = get_moralchoice_prompt(5, context, row['option_a'], row['option_b'],
                                         include_confidence=include_confidence)
        response1 = call_with_rate_limit(prompt1, thinking)

        prompt2 = get_moralchoice_prompt(5, context, row['option_a'], row['option_b'],
                                         previous_response=response1.content,
                                         include_confidence=include_confidence)
        response2 = call_with_rate_limit(prompt2, thinking)

        return {
            'content': response2.content,
            'thinking': response2.thinking,
            'full_response': f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}",
            'input_tokens': response1.input_tokens + response2.input_tokens,
            'output_tokens': response1.output_tokens + response2.output_tokens,
        }
    else:
        prompt = get_moralchoice_prompt(level, context, row['option_a'], row['option_b'],
                                        include_confidence=include_confidence)
        response = call_with_rate_limit(prompt, thinking)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


def run_ethics_experiment():
    """Run full ETHICS experiment."""

    ethics = pd.read_csv("data/ethics_sample.csv")
    if SAMPLE_SIZE:
        # Balance by subscale
        per_subscale = SAMPLE_SIZE // 3
        subscales = []
        for subscale in ['commonsense', 'deontology', 'virtue']:
            subset = ethics[ethics['subscale'] == subscale].head(per_subscale)
            subscales.append(subset)
        ethics = pd.concat(subscales, ignore_index=True)
        print(f"  Using {len(ethics)} items (balanced: {per_subscale} per subscale)")
    results = []

    total_conditions = len(LEVELS) * len(THINKING_CONDITIONS) * N_RUNS
    condition_num = 0

    for run in range(N_RUNS):
        for thinking in THINKING_CONDITIONS:
            for level in LEVELS:
                condition_num += 1
                thinking_label = "ON" if thinking else "OFF"

                print(f"\n[{condition_num}/{total_conditions}] "
                      f"ETHICS Run {run+1}, Level {level}, Thinking {thinking_label}")

                for _, row in tqdm(ethics.iterrows(), total=len(ethics),
                                   desc=f"R{run+1}-L{level}-{thinking_label}"):

                    try:
                        response_data = run_single_item_ethics(row, level, thinking)

                        extracted = extract_ethics_answer(response_data['content'])

                        results.append({
                            'item_id': row['item_id'],
                            'subscale': row['subscale'],
                            'scenario': row['scenario'][:200],  # Truncate for storage
                            'correct_answer': row['label'],
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'response': response_data['full_response'],
                            'thinking_content': response_data['thinking'],
                            'extracted_answer': extracted,
                            'correct': extracted == row['label'] if extracted else None,
                            'response_length': len(response_data['full_response'].split()),
                            'reasoning_markers': count_reasoning_markers(response_data['full_response']),
                            'uncertainty_markers': count_uncertainty_markers(response_data['full_response']),
                            'input_tokens': response_data['input_tokens'],
                            'output_tokens': response_data['output_tokens'],
                            'timestamp': datetime.now().isoformat(),
                        })

                    except Exception as e:
                        print(f"Error on item {row['item_id']}: {e}")
                        results.append({
                            'item_id': row['item_id'],
                            'subscale': row['subscale'],
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                # Save checkpoint after each condition
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(f"results/raw/ethics_checkpoint.csv", index=False)

    return pd.DataFrame(results)


def run_moralchoice_experiment(include_confidence=True):
    """Run full MoralChoice experiment.

    Args:
        include_confidence: If True, prompts ask for both answer and confidence
    """

    mc = pd.read_csv("data/moralchoice_sample.csv")
    if SAMPLE_SIZE:
        # Balance by ambiguity level
        per_level = SAMPLE_SIZE // 2
        low_amb = mc[mc['ambiguity'] == 'low'].head(per_level)
        high_amb = mc[mc['ambiguity'] == 'high'].head(per_level)
        mc = pd.concat([low_amb, high_amb], ignore_index=True)
        print(f"  Using {len(mc)} items (balanced: {len(low_amb)} low, {len(high_amb)} high)")
    results = []

    total_conditions = len(LEVELS) * len(THINKING_CONDITIONS) * N_RUNS
    condition_num = 0

    for run in range(N_RUNS):
        for thinking in THINKING_CONDITIONS:
            for level in LEVELS:
                condition_num += 1
                thinking_label = "ON" if thinking else "OFF"

                print(f"\n[{condition_num}/{total_conditions}] "
                      f"MoralChoice Run {run+1}, Level {level}, Thinking {thinking_label}")

                for _, row in tqdm(mc.iterrows(), total=len(mc),
                                   desc=f"R{run+1}-L{level}-{thinking_label}"):

                    try:
                        response_data = run_single_item_moralchoice(row, level, thinking,
                                                                     include_confidence=include_confidence)

                        # Extract both answer and confidence
                        extraction = extract_moralchoice_with_confidence(response_data['content'])

                        results.append({
                            'item_id': row['item_id'],
                            'context': row['context'][:100],
                            'option_a': row['option_a'][:100],
                            'option_b': row['option_b'][:100],
                            'ambiguity': row.get('ambiguity', None),
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'response': response_data['full_response'],
                            'thinking_content': response_data['thinking'],
                            'extracted_answer': extraction['answer'],
                            'confidence': extraction['confidence'],
                            'confidence_category': extraction['confidence_category'],
                            'response_length': len(response_data['full_response'].split()),
                            'reasoning_markers': count_reasoning_markers(response_data['full_response']),
                            'uncertainty_markers': count_uncertainty_markers(response_data['full_response']),
                            'input_tokens': response_data['input_tokens'],
                            'output_tokens': response_data['output_tokens'],
                            'timestamp': datetime.now().isoformat(),
                        })

                    except Exception as e:
                        print(f"Error on item {row['item_id']}: {e}")
                        results.append({
                            'item_id': row['item_id'],
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                # Checkpoint
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(f"results/raw/moralchoice_checkpoint.csv", index=False)

    return pd.DataFrame(results)


def main():
    start_time = datetime.now()
    print("=" * 60)
    print("STARTING MAIN EXPERIMENT")
    print("=" * 60)
    print(f"Start time: {start_time}")
    print(f"Model: {config.MODEL}")
    print(f"Levels: {LEVELS}")
    print(f"Thinking conditions: {THINKING_CONDITIONS}")
    print(f"Runs: {N_RUNS}")

    # Calculate total API calls
    ethics_items = min(SAMPLE_SIZE, len(pd.read_csv("data/ethics_sample.csv"))) if SAMPLE_SIZE else len(pd.read_csv("data/ethics_sample.csv"))
    mc_items = min(SAMPLE_SIZE, len(pd.read_csv("data/moralchoice_sample.csv"))) if SAMPLE_SIZE else len(pd.read_csv("data/moralchoice_sample.csv"))

    # Level 5 requires 2 calls per item
    ethics_calls = ethics_items * (5 * 1 + 1 * 2) * 2 * N_RUNS  # 5 single-call levels + 1 two-call level
    mc_calls = mc_items * (5 * 1 + 1 * 2) * 2 * N_RUNS

    print(f"\nDataset sizes:")
    print(f"  ETHICS: {ethics_items} items")
    print(f"  MoralChoice: {mc_items} items")
    print(f"\nEstimated API calls: ~{ethics_calls + mc_calls}")

    # Create output directories
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/processed").mkdir(parents=True, exist_ok=True)

    # Run ETHICS
    print("\n" + "=" * 60)
    print("RUNNING ETHICS EXPERIMENT")
    print("=" * 60)
    ethics_results = run_ethics_experiment()
    ethics_results.to_csv("results/processed/ethics_results.csv", index=False)
    print(f"\nETHICS complete: {len(ethics_results)} observations")

    # Run MoralChoice
    print("\n" + "=" * 60)
    print("RUNNING MORALCHOICE EXPERIMENT")
    print("=" * 60)
    mc_results = run_moralchoice_experiment()
    mc_results.to_csv("results/processed/moralchoice_results.csv", index=False)
    print(f"\nMoralChoice complete: {len(mc_results)} observations")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration}")
    print(f"ETHICS results: results/processed/ethics_results.csv")
    print(f"MoralChoice results: results/processed/moralchoice_results.csv")


if __name__ == "__main__":
    main()
