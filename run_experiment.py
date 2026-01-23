"""Run full experiment for all benchmarks (sync or async mode)."""

import argparse
import asyncio
import pandas as pd
import os
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Optional

# Load environment variables from .env
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            match = re.match(r'(\w+)\s*=\s*["\']?([^"\']+)["\']?', line)
            if match:
                os.environ[match.group(1)] = match.group(2)

from prompts import get_ethics_prompt, get_moralchoice_prompt, get_morables_prompt
from src.api import (
    call_with_rate_limit,
    call_with_rate_limit_async,
    reset_rate_limiter,
    get_rate_stats,
    APIResponse
)
from src.extraction import (
    extract_ethics_answer,
    extract_moralchoice_answer,
    extract_morables_answer,
    extract_moralchoice_with_confidence,
    extract_confidence_score,
    categorize_confidence,
    count_reasoning_markers,
    count_uncertainty_markers
)
import config


# =============================================================================
# EXPERIMENT SETTINGS (defaults, can be overridden via CLI)
# =============================================================================

LEVELS = [0, 1, 2, 3, 4, 5]
THINKING_CONDITIONS = [False, True]
N_RUNS = config.N_RUNS
SAMPLE_SIZE = 100  # Set to None for full dataset
INCLUDE_CONFIDENCE = True


# =============================================================================
# SINGLE ITEM RUNNERS (SYNC)
# =============================================================================

def run_single_item_ethics(row, level, thinking, include_confidence=True):
    """Run single ETHICS item at given condition (sync)."""
    # Use reduced token limit for Level 0 (only when thinking is disabled)
    max_tokens = config.MAX_TOKENS_LEVEL_0 if (level == 0 and not thinking) else None

    if level == 5:
        prompt1 = get_ethics_prompt(5, row['scenario'], include_confidence=include_confidence)
        response1 = call_with_rate_limit(prompt1, thinking)

        prompt2 = get_ethics_prompt(5, row['scenario'], response1.content,
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
        prompt = get_ethics_prompt(level, row['scenario'], include_confidence=include_confidence)
        response = call_with_rate_limit(prompt, thinking, max_tokens_override=max_tokens)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


def run_single_item_moralchoice(row, level, thinking, include_confidence=True):
    """Run single MoralChoice item at given condition (sync)."""
    context = row.get('context', '')
    # Use reduced token limit for Level 0 (only when thinking is disabled)
    max_tokens = config.MAX_TOKENS_LEVEL_0 if (level == 0 and not thinking) else None

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
        response = call_with_rate_limit(prompt, thinking, max_tokens_override=max_tokens)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


def run_single_item_morables(row, level, thinking, include_confidence=True):
    """Run single MORABLES item at given condition (sync)."""
    options = [row['option_a'], row['option_b'], row['option_c'],
               row['option_d'], row['option_e']]
    # Use reduced token limit for Level 0 (only when thinking is disabled)
    max_tokens = config.MAX_TOKENS_LEVEL_0 if (level == 0 and not thinking) else None

    if level == 5:
        prompt1 = get_morables_prompt(5, row['fable'], options, include_confidence=include_confidence)
        response1 = call_with_rate_limit(prompt1, thinking)

        prompt2 = get_morables_prompt(5, row['fable'], options,
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
        prompt = get_morables_prompt(level, row['fable'], options, include_confidence=include_confidence)
        response = call_with_rate_limit(prompt, thinking, max_tokens_override=max_tokens)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


# =============================================================================
# SINGLE ITEM RUNNERS (ASYNC)
# =============================================================================

async def run_single_item_ethics_async(row, level, thinking, include_confidence=True):
    """Run single ETHICS item at given condition (async)."""
    # Use reduced token limit for Level 0 (only when thinking is disabled)
    max_tokens = config.MAX_TOKENS_LEVEL_0 if (level == 0 and not thinking) else None

    if level == 5:
        prompt1 = get_ethics_prompt(5, row['scenario'], include_confidence=include_confidence)
        response1 = await call_with_rate_limit_async(prompt1, thinking)

        prompt2 = get_ethics_prompt(5, row['scenario'],
                                    previous_response=response1.content,
                                    include_confidence=include_confidence)
        response2 = await call_with_rate_limit_async(prompt2, thinking)

        return {
            'content': response2.content,
            'thinking': response2.thinking,
            'full_response': f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}",
            'input_tokens': response1.input_tokens + response2.input_tokens,
            'output_tokens': response1.output_tokens + response2.output_tokens,
        }
    else:
        prompt = get_ethics_prompt(level, row['scenario'], include_confidence=include_confidence)
        response = await call_with_rate_limit_async(prompt, thinking, max_tokens_override=max_tokens)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


async def run_single_item_moralchoice_async(row, level, thinking, include_confidence=True):
    """Run single MoralChoice item at given condition (async)."""
    # Use reduced token limit for Level 0 (only when thinking is disabled)
    max_tokens = config.MAX_TOKENS_LEVEL_0 if (level == 0 and not thinking) else None

    if level == 5:
        prompt1 = get_moralchoice_prompt(5, row['context'], row['option_a'], row['option_b'],
                                         include_confidence=include_confidence)
        response1 = await call_with_rate_limit_async(prompt1, thinking)

        prompt2 = get_moralchoice_prompt(5, row['context'], row['option_a'], row['option_b'],
                                         previous_response=response1.content,
                                         include_confidence=include_confidence)
        response2 = await call_with_rate_limit_async(prompt2, thinking)

        return {
            'content': response2.content,
            'thinking': response2.thinking,
            'full_response': f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}",
            'input_tokens': response1.input_tokens + response2.input_tokens,
            'output_tokens': response1.output_tokens + response2.output_tokens,
        }
    else:
        prompt = get_moralchoice_prompt(level, row['context'], row['option_a'], row['option_b'],
                                        include_confidence=include_confidence)
        response = await call_with_rate_limit_async(prompt, thinking, max_tokens_override=max_tokens)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


async def run_single_item_morables_async(row, level, thinking, include_confidence=True):
    """Run single MORABLES item at given condition (async)."""
    options = [row['option_a'], row['option_b'], row['option_c'],
               row['option_d'], row['option_e']]
    # Use reduced token limit for Level 0 (only when thinking is disabled)
    max_tokens = config.MAX_TOKENS_LEVEL_0 if (level == 0 and not thinking) else None

    if level == 5:
        prompt1 = get_morables_prompt(5, row['fable'], options, include_confidence=include_confidence)
        response1 = await call_with_rate_limit_async(prompt1, thinking)

        prompt2 = get_morables_prompt(5, row['fable'], options,
                                      previous_response=response1.content,
                                      include_confidence=include_confidence)
        response2 = await call_with_rate_limit_async(prompt2, thinking)

        return {
            'content': response2.content,
            'thinking': response2.thinking,
            'full_response': f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}",
            'input_tokens': response1.input_tokens + response2.input_tokens,
            'output_tokens': response1.output_tokens + response2.output_tokens,
        }
    else:
        prompt = get_morables_prompt(level, row['fable'], options, include_confidence=include_confidence)
        response = await call_with_rate_limit_async(prompt, thinking, max_tokens_override=max_tokens)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


# =============================================================================
# RESULT BUILDERS
# =============================================================================

def build_ethics_result(row, level, thinking, run, response_data, include_confidence):
    """Build result dict for ETHICS item."""
    extracted = extract_ethics_answer(response_data['content'])
    confidence = extract_confidence_score(response_data['content']) if include_confidence else None
    correct = (extracted == row['label']) if extracted else None

    return {
        'item_id': row['item_id'],
        'subscale': row.get('subscale', ''),
        'scenario': row['scenario'][:200],
        'label': row['label'],
        'level': level,
        'thinking': thinking,
        'run': run,
        'response': response_data['full_response'],
        'thinking_content': response_data['thinking'],
        'extracted_answer': extracted,
        'correct': correct,
        'confidence': confidence,
        'confidence_category': categorize_confidence(confidence),
        'response_length': len(response_data['full_response'].split()),
        'reasoning_markers': count_reasoning_markers(response_data['full_response']),
        'uncertainty_markers': count_uncertainty_markers(response_data['full_response']),
        'input_tokens': response_data['input_tokens'],
        'output_tokens': response_data['output_tokens'],
        'timestamp': datetime.now().isoformat(),
    }


def build_moralchoice_result(row, level, thinking, run, response_data, include_confidence):
    """Build result dict for MoralChoice item."""
    extraction = extract_moralchoice_with_confidence(response_data['content'])

    return {
        'item_id': row['item_id'],
        'context': row['context'][:200],
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
    }


def build_morables_result(row, level, thinking, run, response_data, include_confidence):
    """Build result dict for MORABLES item."""
    extracted = extract_morables_answer(response_data['content'])
    confidence = extract_confidence_score(response_data['content']) if include_confidence else None

    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    extracted_idx = letter_to_idx.get(extracted)
    correct = extracted_idx == row['correct_idx'] if extracted_idx is not None else None

    return {
        'item_id': row['item_id'],
        'fable': row['fable'][:200],
        'correct_idx': row['correct_idx'],
        'correct_answer': ['A', 'B', 'C', 'D', 'E'][row['correct_idx']],
        'level': level,
        'thinking': thinking,
        'run': run,
        'response': response_data['full_response'],
        'thinking_content': response_data['thinking'],
        'extracted_answer': extracted,
        'correct': correct,
        'confidence': confidence,
        'confidence_category': categorize_confidence(confidence),
        'response_length': len(response_data['full_response'].split()),
        'reasoning_markers': count_reasoning_markers(response_data['full_response']),
        'uncertainty_markers': count_uncertainty_markers(response_data['full_response']),
        'input_tokens': response_data['input_tokens'],
        'output_tokens': response_data['output_tokens'],
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# SYNC EXPERIMENT RUNNERS
# =============================================================================

def run_ethics_experiment(sample_size=None, include_confidence=True):
    """Run full ETHICS experiment (sync)."""
    ethics = pd.read_csv("data/ethics_sample.csv")

    if sample_size:
        per_subscale = sample_size // 3
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
                        response_data = run_single_item_ethics(row, level, thinking, include_confidence)
                        results.append(build_ethics_result(row, level, thinking, run, response_data, include_confidence))
                    except Exception as e:
                        print(f"Error on item {row['item_id']}: {e}")
                        results.append({
                            'item_id': row['item_id'],
                            'subscale': row.get('subscale', ''),
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                # Checkpoint
                pd.DataFrame(results).to_csv("results/raw/ethics_checkpoint.csv", index=False)

    return pd.DataFrame(results)


def run_moralchoice_experiment(sample_size=None, include_confidence=True):
    """Run full MoralChoice experiment (sync)."""
    mc = pd.read_csv("data/moralchoice_sample.csv")

    if sample_size:
        per_level = sample_size // 2
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
                        response_data = run_single_item_moralchoice(row, level, thinking, include_confidence)
                        results.append(build_moralchoice_result(row, level, thinking, run, response_data, include_confidence))
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
                pd.DataFrame(results).to_csv("results/raw/moralchoice_checkpoint.csv", index=False)

    return pd.DataFrame(results)


def run_morables_experiment(sample_size=None, include_confidence=True):
    """Run full MORABLES experiment (sync)."""
    data_path = "data/morables/morables_sample.csv"
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found. Run 'python prepare_data.py' first.")
        return None

    morables = pd.read_csv(data_path)

    if sample_size and len(morables) > sample_size:
        morables = morables.sample(n=sample_size, random_state=config.RANDOM_SEED)
        print(f"  Using {len(morables)} items (sampled)")
    else:
        print(f"  Using {len(morables)} items")

    results = []
    total_conditions = len(LEVELS) * len(THINKING_CONDITIONS) * N_RUNS
    condition_num = 0

    for run in range(N_RUNS):
        for thinking in THINKING_CONDITIONS:
            for level in LEVELS:
                condition_num += 1
                thinking_label = "ON" if thinking else "OFF"

                print(f"\n[{condition_num}/{total_conditions}] "
                      f"MORABLES Run {run+1}, Level {level}, Thinking {thinking_label}")

                for _, row in tqdm(morables.iterrows(), total=len(morables),
                                   desc=f"R{run+1}-L{level}-{thinking_label}"):
                    try:
                        response_data = run_single_item_morables(row, level, thinking, include_confidence)
                        results.append(build_morables_result(row, level, thinking, run, response_data, include_confidence))
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
                pd.DataFrame(results).to_csv("results/raw/morables_checkpoint.csv", index=False)

    return pd.DataFrame(results)


# =============================================================================
# ASYNC EXPERIMENT RUNNERS
# =============================================================================

async def run_ethics_experiment_async(results_queue: asyncio.Queue, sample_size=None, include_confidence=True):
    """Run ETHICS experiment asynchronously."""
    data_path = "data/ethics_sample.csv"
    if not Path(data_path).exists():
        print(f"ETHICS: {data_path} not found. Skipping.")
        return []

    ethics = pd.read_csv(data_path)

    if sample_size:
        per_subscale = sample_size // 3
        subscales = []
        for subscale in ['commonsense', 'deontology', 'virtue']:
            subset = ethics[ethics['subscale'] == subscale].head(per_subscale)
            subscales.append(subset)
        ethics = pd.concat(subscales, ignore_index=True)

    print(f"ETHICS: Starting with {len(ethics)} items")
    results = []

    for run in range(N_RUNS):
        for thinking in THINKING_CONDITIONS:
            for level in LEVELS:
                thinking_label = "ON" if thinking else "OFF"

                for idx, row in ethics.iterrows():
                    try:
                        response_data = await run_single_item_ethics_async(row, level, thinking, include_confidence)
                        result = build_ethics_result(row, level, thinking, run, response_data, include_confidence)
                        result['benchmark'] = 'ethics'
                        results.append(result)
                        await results_queue.put(('ethics', result))
                    except Exception as e:
                        print(f"ETHICS error on {row['item_id']}: {e}")
                        results.append({
                            'benchmark': 'ethics',
                            'item_id': row['item_id'],
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                stats = get_rate_stats()
                print(f"ETHICS: L{level} T-{thinking_label} done | Rate: {stats['rate']}/min")

    print(f"ETHICS: Complete with {len(results)} results")
    return results


async def run_moralchoice_experiment_async(results_queue: asyncio.Queue, sample_size=None, include_confidence=True):
    """Run MoralChoice experiment asynchronously."""
    data_path = "data/moralchoice_sample.csv"
    if not Path(data_path).exists():
        print(f"MoralChoice: {data_path} not found. Skipping.")
        return []

    mc = pd.read_csv(data_path)

    if sample_size:
        per_level = sample_size // 2
        low_amb = mc[mc['ambiguity'] == 'low'].head(per_level)
        high_amb = mc[mc['ambiguity'] == 'high'].head(per_level)
        mc = pd.concat([low_amb, high_amb], ignore_index=True)

    print(f"MoralChoice: Starting with {len(mc)} items")
    results = []

    for run in range(N_RUNS):
        for thinking in THINKING_CONDITIONS:
            for level in LEVELS:
                thinking_label = "ON" if thinking else "OFF"

                for idx, row in mc.iterrows():
                    try:
                        response_data = await run_single_item_moralchoice_async(row, level, thinking, include_confidence)
                        result = build_moralchoice_result(row, level, thinking, run, response_data, include_confidence)
                        result['benchmark'] = 'moralchoice'
                        results.append(result)
                        await results_queue.put(('moralchoice', result))
                    except Exception as e:
                        print(f"MoralChoice error on {row['item_id']}: {e}")
                        results.append({
                            'benchmark': 'moralchoice',
                            'item_id': row['item_id'],
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                stats = get_rate_stats()
                print(f"MoralChoice: L{level} T-{thinking_label} done | Rate: {stats['rate']}/min")

    print(f"MoralChoice: Complete with {len(results)} results")
    return results


async def run_morables_experiment_async(results_queue: asyncio.Queue, sample_size=None, include_confidence=True):
    """Run MORABLES experiment asynchronously."""
    data_path = "data/morables/morables_sample.csv"
    if not Path(data_path).exists():
        print(f"MORABLES: {data_path} not found. Skipping.")
        return []

    morables = pd.read_csv(data_path)
    if sample_size and len(morables) > sample_size:
        morables = morables.sample(n=sample_size, random_state=config.RANDOM_SEED)

    print(f"MORABLES: Starting with {len(morables)} items")
    results = []

    for run in range(N_RUNS):
        for thinking in THINKING_CONDITIONS:
            for level in LEVELS:
                thinking_label = "ON" if thinking else "OFF"

                for idx, row in morables.iterrows():
                    try:
                        response_data = await run_single_item_morables_async(row, level, thinking, include_confidence)
                        result = build_morables_result(row, level, thinking, run, response_data, include_confidence)
                        result['benchmark'] = 'morables'
                        results.append(result)
                        await results_queue.put(('morables', result))
                    except Exception as e:
                        print(f"MORABLES error on {row['item_id']}: {e}")
                        results.append({
                            'benchmark': 'morables',
                            'item_id': row['item_id'],
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                stats = get_rate_stats()
                print(f"MORABLES: L{level} T-{thinking_label} done | Rate: {stats['rate']}/min")

    print(f"MORABLES: Complete with {len(results)} results")
    return results


async def checkpoint_writer(results_queue: asyncio.Queue, checkpoint_interval: int = 50):
    """Background task to write checkpoints as results come in."""
    all_results = {'ethics': [], 'moralchoice': [], 'morables': []}
    count = 0

    while True:
        try:
            benchmark, result = await asyncio.wait_for(results_queue.get(), timeout=1.0)
            all_results[benchmark].append(result)
            count += 1

            if count % checkpoint_interval == 0:
                for bm, results in all_results.items():
                    if results:
                        pd.DataFrame(results).to_csv(f"results/raw/{bm}_checkpoint.csv", index=False)
                print(f"  [Checkpoint] {count} results saved")

        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            for bm, results in all_results.items():
                if results:
                    pd.DataFrame(results).to_csv(f"results/raw/{bm}_checkpoint.csv", index=False)
            print(f"  [Final checkpoint] {count} total results saved")
            break


# =============================================================================
# MAIN RUNNERS
# =============================================================================

def run_sync(args):
    """Run experiments synchronously (sequential)."""
    start_time = datetime.now()

    print("=" * 60)
    print("STARTING EXPERIMENT (Sync Mode)")
    print("=" * 60)
    print(f"Start time: {start_time}")
    print(f"Model: {config.MODEL}")
    print(f"Levels: {LEVELS}")
    print(f"Thinking conditions: {THINKING_CONDITIONS}")
    print(f"Runs: {N_RUNS}")
    print(f"Sample size: {args.sample_size}")

    # Create output directories
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/processed").mkdir(parents=True, exist_ok=True)

    # Run ETHICS
    if args.ethics:
        print("\n" + "=" * 60)
        print("RUNNING ETHICS EXPERIMENT")
        print("=" * 60)
        ethics_results = run_ethics_experiment(args.sample_size, args.confidence)
        ethics_results.to_csv("results/processed/ethics_results.csv", index=False)
        print(f"\nETHICS complete: {len(ethics_results)} observations")

    # Run MoralChoice
    if args.moralchoice:
        print("\n" + "=" * 60)
        print("RUNNING MORALCHOICE EXPERIMENT")
        print("=" * 60)
        mc_results = run_moralchoice_experiment(args.sample_size, args.confidence)
        mc_results.to_csv("results/processed/moralchoice_results.csv", index=False)
        print(f"\nMoralChoice complete: {len(mc_results)} observations")

    # Run MORABLES
    if args.morables:
        print("\n" + "=" * 60)
        print("RUNNING MORABLES EXPERIMENT")
        print("=" * 60)
        morables_results = run_morables_experiment(args.sample_size, args.confidence)
        if morables_results is not None:
            morables_results.to_csv("results/processed/morables_results.csv", index=False)
            print(f"\nMORABLES complete: {len(morables_results)} observations")

            valid_results = morables_results[morables_results['correct'].notna()]
            if len(valid_results) > 0:
                print(f"  Overall accuracy: {valid_results['correct'].mean():.3f}")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration}")


async def run_async(args):
    """Run experiments asynchronously (parallel datasets)."""
    start_time = datetime.now()

    print("=" * 60)
    print("STARTING EXPERIMENT (Async Mode - Parallel Datasets)")
    print("=" * 60)
    print(f"Start time: {start_time}")
    print(f"Model: {config.MODEL}")
    print(f"Levels: {LEVELS}")
    print(f"Thinking conditions: {THINKING_CONDITIONS}")
    print(f"Runs: {N_RUNS}")
    print(f"Sample size: {args.sample_size}")
    print(f"Rate limit: {config.CALLS_PER_MINUTE}/min")
    print()

    # Create output directories
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/processed").mkdir(parents=True, exist_ok=True)

    # Reset rate limiter
    reset_rate_limiter()

    # Results queue for checkpoint writing
    results_queue = asyncio.Queue()

    # Start checkpoint writer
    checkpoint_task = asyncio.create_task(checkpoint_writer(results_queue))

    # Build list of experiment tasks
    tasks = []
    if args.ethics:
        tasks.append(run_ethics_experiment_async(results_queue, args.sample_size, args.confidence))
    if args.moralchoice:
        tasks.append(run_moralchoice_experiment_async(results_queue, args.sample_size, args.confidence))
    if args.morables:
        tasks.append(run_morables_experiment_async(results_queue, args.sample_size, args.confidence))

    print(f"Running {len(tasks)} benchmark(s) in parallel...")
    print()

    # Run all experiments concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Stop checkpoint writer
    checkpoint_task.cancel()
    try:
        await checkpoint_task
    except asyncio.CancelledError:
        pass

    # Process results
    task_idx = 0

    if args.ethics:
        if isinstance(results[task_idx], Exception):
            print(f"ETHICS failed with error: {results[task_idx]}")
        else:
            ethics_results = results[task_idx]
            if ethics_results:
                pd.DataFrame(ethics_results).to_csv("results/processed/ethics_results.csv", index=False)
                print(f"ETHICS: Saved {len(ethics_results)} results")
        task_idx += 1

    if args.moralchoice:
        if isinstance(results[task_idx], Exception):
            print(f"MoralChoice failed with error: {results[task_idx]}")
        else:
            mc_results = results[task_idx]
            if mc_results:
                pd.DataFrame(mc_results).to_csv("results/processed/moralchoice_results.csv", index=False)
                print(f"MoralChoice: Saved {len(mc_results)} results")
        task_idx += 1

    if args.morables:
        if isinstance(results[task_idx], Exception):
            print(f"MORABLES failed with error: {results[task_idx]}")
        else:
            morables_results = results[task_idx]
            if morables_results:
                pd.DataFrame(morables_results).to_csv("results/processed/morables_results.csv", index=False)
                print(f"MORABLES: Saved {len(morables_results)} results")

                df = pd.DataFrame(morables_results)
                valid = df[df['correct'].notna()]
                if len(valid) > 0:
                    print(f"  MORABLES accuracy: {valid['correct'].mean():.3f}")

    end_time = datetime.now()
    duration = end_time - start_time
    stats = get_rate_stats()

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration}")
    print(f"Total API calls: {stats['calls']}")
    print(f"Effective rate: {stats['rate']} calls/min")


def main():
    parser = argparse.ArgumentParser(
        description="Run Variable Reflection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py                    # Run all benchmarks (sync)
  python run_experiment.py --async            # Run all benchmarks (async, ~2.5x faster)
  python run_experiment.py --ethics           # Run only ETHICS
  python run_experiment.py --async --morables # Run only MORABLES (async)
  python run_experiment.py --sample 50        # Use 50 items per benchmark
        """
    )

    parser.add_argument('--async', dest='use_async', action='store_true',
                        help='Use async mode for parallel dataset execution (~2.5x faster)')
    parser.add_argument('--sample', dest='sample_size', type=int, default=SAMPLE_SIZE,
                        help=f'Sample size per benchmark (default: {SAMPLE_SIZE}, use 0 for full)')
    parser.add_argument('--no-confidence', dest='confidence', action='store_false',
                        help='Disable confidence scoring')

    # Benchmark selection
    parser.add_argument('--ethics', action='store_true', help='Run ETHICS benchmark')
    parser.add_argument('--moralchoice', action='store_true', help='Run MoralChoice benchmark')
    parser.add_argument('--morables', action='store_true', help='Run MORABLES benchmark')

    args = parser.parse_args()

    # If no benchmarks specified, run all
    if not (args.ethics or args.moralchoice or args.morables):
        args.ethics = True
        args.moralchoice = True
        args.morables = True

    # Handle sample_size=0 as None (full dataset)
    if args.sample_size == 0:
        args.sample_size = None

    if args.use_async:
        asyncio.run(run_async(args))
    else:
        run_sync(args)


if __name__ == "__main__":
    main()
