"""Run full experiment for all benchmarks (sync or async mode)."""

import argparse
import asyncio
import pandas as pd
import os
import re
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
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

LEVELS = [0, 2, 4, 5]
THINKING_CONDITIONS = [False, True]
N_RUNS = config.N_RUNS
SAMPLE_SIZE = 100  # Set to None for full dataset
INCLUDE_CONFIDENCE = True


# =============================================================================
# SINGLE ITEM RUNNERS (Unified async implementation with sync wrappers)
# =============================================================================

def _build_response_data(response1, response2=None):
    """Build standardized response data dict from API response(s)."""
    if response2 is not None:
        # Two-pass (Level 5)
        return {
            'content': response2.content,
            'thinking': response2.thinking,
            'full_response': f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}",
            'input_tokens': response1.input_tokens + response2.input_tokens,
            'output_tokens': response1.output_tokens + response2.output_tokens,
        }
    else:
        # Single pass
        return {
            'content': response1.content,
            'thinking': response1.thinking,
            'full_response': response1.content,
            'input_tokens': response1.input_tokens,
            'output_tokens': response1.output_tokens,
        }


async def _run_single_item_async(prompt_fn, level, thinking, include_confidence, **prompt_kwargs):
    """
    Generic async single-item runner.

    Args:
        prompt_fn: Prompt generator function (get_ethics_prompt, get_moralchoice_prompt, etc.)
        level: Reflection level (0-5)
        thinking: Whether extended thinking is enabled
        include_confidence: Whether to include confidence in prompt
        **prompt_kwargs: Additional kwargs passed to prompt_fn (scenario, context, options, etc.)

    Returns:
        Response data dict with content, thinking, full_response, and token counts
    """
    max_tokens = config.MAX_TOKENS_LEVEL_0 if (level == 0 and not thinking) else None

    if level == 5:
        # Two-pass reflection
        prompt1 = prompt_fn(5, **prompt_kwargs, include_confidence=include_confidence)
        response1 = await call_with_rate_limit_async(prompt1, thinking)

        prompt2 = prompt_fn(5, **prompt_kwargs, previous_response=response1.content,
                           include_confidence=include_confidence)
        response2 = await call_with_rate_limit_async(prompt2, thinking)

        return _build_response_data(response1, response2)
    else:
        prompt = prompt_fn(level, **prompt_kwargs, include_confidence=include_confidence)
        response = await call_with_rate_limit_async(prompt, thinking, max_tokens_override=max_tokens)

        return _build_response_data(response)


def _run_single_item_sync(prompt_fn, level, thinking, include_confidence, **prompt_kwargs):
    """
    Generic sync single-item runner.

    Args:
        prompt_fn: Prompt generator function (get_ethics_prompt, get_moralchoice_prompt, etc.)
        level: Reflection level (0-5)
        thinking: Whether extended thinking is enabled
        include_confidence: Whether to include confidence in prompt
        **prompt_kwargs: Additional kwargs passed to prompt_fn

    Returns:
        Response data dict with content, thinking, full_response, and token counts
    """
    max_tokens = config.MAX_TOKENS_LEVEL_0 if (level == 0 and not thinking) else None

    if level == 5:
        # Two-pass reflection
        prompt1 = prompt_fn(5, **prompt_kwargs, include_confidence=include_confidence)
        response1 = call_with_rate_limit(prompt1, thinking)

        prompt2 = prompt_fn(5, **prompt_kwargs, previous_response=response1.content,
                           include_confidence=include_confidence)
        response2 = call_with_rate_limit(prompt2, thinking)

        return _build_response_data(response1, response2)
    else:
        prompt = prompt_fn(level, **prompt_kwargs, include_confidence=include_confidence)
        response = call_with_rate_limit(prompt, thinking, max_tokens_override=max_tokens)

        return _build_response_data(response)


# Benchmark-specific runners (async) - thin wrappers around generic runner

async def run_single_item_ethics_async(row, level, thinking, include_confidence=True):
    """Run single ETHICS item at given condition (async)."""
    return await _run_single_item_async(
        get_ethics_prompt, level, thinking, include_confidence,
        scenario=row['scenario']
    )


async def run_single_item_moralchoice_async(row, level, thinking, include_confidence=True):
    """Run single MoralChoice item at given condition (async)."""
    return await _run_single_item_async(
        get_moralchoice_prompt, level, thinking, include_confidence,
        context=row['context'], option_a=row['option_a'], option_b=row['option_b']
    )


async def run_single_item_morables_async(row, level, thinking, include_confidence=True):
    """Run single MORABLES item at given condition (async)."""
    options = [row['option_a'], row['option_b'], row['option_c'],
               row['option_d'], row['option_e']]
    return await _run_single_item_async(
        get_morables_prompt, level, thinking, include_confidence,
        fable=row['fable'], options=options
    )


# Benchmark-specific runners (sync) - thin wrappers for backwards compatibility

def run_single_item_ethics(row, level, thinking, include_confidence=True):
    """Run single ETHICS item at given condition (sync)."""
    return _run_single_item_sync(
        get_ethics_prompt, level, thinking, include_confidence,
        scenario=row['scenario']
    )


def run_single_item_moralchoice(row, level, thinking, include_confidence=True):
    """Run single MoralChoice item at given condition (sync)."""
    return _run_single_item_sync(
        get_moralchoice_prompt, level, thinking, include_confidence,
        context=row.get('context', ''), option_a=row['option_a'], option_b=row['option_b']
    )


def run_single_item_morables(row, level, thinking, include_confidence=True):
    """Run single MORABLES item at given condition (sync)."""
    options = [row['option_a'], row['option_b'], row['option_c'],
               row['option_d'], row['option_e']]
    return _run_single_item_sync(
        get_morables_prompt, level, thinking, include_confidence,
        fable=row['fable'], options=options
    )


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
# DATA LOADING HELPERS (shared by sync and async runners)
# =============================================================================

def load_ethics_data(sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Load and optionally sample ETHICS data with stratification by subscale."""
    data_path = "data/ethics_sample.csv"
    if not Path(data_path).exists():
        print(f"ETHICS: {data_path} not found. Skipping.")
        return None

    ethics = pd.read_csv(data_path)

    if sample_size:
        per_subscale = sample_size // 3
        subscales = []
        for subscale in ['commonsense', 'deontology', 'virtue']:
            stratum = ethics[ethics['subscale'] == subscale]
            subset = stratum.sample(n=min(per_subscale, len(stratum)), random_state=config.RANDOM_SEED)
            subscales.append(subset)
        ethics = pd.concat(subscales, ignore_index=True)
        print(f"  Using {len(ethics)} items (stratified: {per_subscale} per subscale)")
    else:
        print(f"  Using {len(ethics)} items")

    return ethics


def load_moralchoice_data(sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Load and optionally sample MoralChoice data with stratification by ambiguity."""
    data_path = "data/moralchoice_sample.csv"
    if not Path(data_path).exists():
        print(f"MoralChoice: {data_path} not found. Skipping.")
        return None

    mc = pd.read_csv(data_path)

    if sample_size:
        per_level = sample_size // 2
        low_stratum = mc[mc['ambiguity'] == 'low']
        high_stratum = mc[mc['ambiguity'] == 'high']
        low_amb = low_stratum.sample(n=min(per_level, len(low_stratum)), random_state=config.RANDOM_SEED)
        high_amb = high_stratum.sample(n=min(per_level, len(high_stratum)), random_state=config.RANDOM_SEED)
        mc = pd.concat([low_amb, high_amb], ignore_index=True)
        print(f"  Using {len(mc)} items (stratified: {len(low_amb)} low, {len(high_amb)} high)")
    else:
        print(f"  Using {len(mc)} items")

    return mc


def load_morables_data(sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Load and optionally sample MORABLES data."""
    data_path = "data/morables/morables_sample.csv"
    if not Path(data_path).exists():
        print(f"MORABLES: {data_path} not found. Skipping.")
        return None

    morables = pd.read_csv(data_path)

    if sample_size and len(morables) > sample_size:
        morables = morables.sample(n=sample_size, random_state=config.RANDOM_SEED)
        print(f"  Using {len(morables)} items (sampled)")
    else:
        print(f"  Using {len(morables)} items")

    return morables


def _build_error_result(row, level, thinking, run, error, extra_fields=None):
    """Build standardized error result dict."""
    result = {
        'item_id': row['item_id'],
        'level': level,
        'thinking': thinking,
        'run': run,
        'error': str(error),
        'timestamp': datetime.now().isoformat(),
    }
    if extra_fields:
        result.update(extra_fields)
    return result


# =============================================================================
# SYNC EXPERIMENT RUNNERS
# =============================================================================

def _run_experiment_sync(
    benchmark_name: str,
    data: pd.DataFrame,
    checkpoint_path: str,
    item_runner,
    result_builder,
    error_extra_fields_fn,
    include_confidence: bool = True
) -> pd.DataFrame:
    """
    Generic sync experiment runner.

    Args:
        benchmark_name: Name for logging (e.g., "ETHICS")
        data: DataFrame with items to process
        checkpoint_path: Path for checkpoint file
        item_runner: Function(row, level, thinking, include_confidence) -> response_data
        result_builder: Function(row, level, thinking, run, response_data, include_confidence) -> result dict
        error_extra_fields_fn: Function(row) -> dict of extra fields for error results
        include_confidence: Whether to include confidence scoring
    """
    results = []
    total_conditions = len(LEVELS) * len(THINKING_CONDITIONS) * N_RUNS
    condition_num = 0

    for run in range(N_RUNS):
        for thinking in THINKING_CONDITIONS:
            for level in LEVELS:
                condition_num += 1
                thinking_label = "ON" if thinking else "OFF"

                print(f"\n[{condition_num}/{total_conditions}] "
                      f"{benchmark_name} Run {run+1}, Level {level}, Thinking {thinking_label}")

                for _, row in tqdm(data.iterrows(), total=len(data),
                                   desc=f"R{run+1}-L{level}-{thinking_label}"):
                    try:
                        response_data = item_runner(row, level, thinking, include_confidence)
                        results.append(result_builder(row, level, thinking, run, response_data, include_confidence))
                    except Exception as e:
                        print(f"Error on item {row['item_id']}: {e}")
                        extra = error_extra_fields_fn(row) if error_extra_fields_fn else {}
                        results.append(_build_error_result(row, level, thinking, run, e, extra))

                # Checkpoint after each condition
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(results)


def run_ethics_experiment(sample_size=None, include_confidence=True):
    """Run full ETHICS experiment (sync)."""
    ethics = load_ethics_data(sample_size)
    if ethics is None:
        return None

    return _run_experiment_sync(
        benchmark_name="ETHICS",
        data=ethics,
        checkpoint_path="results/raw/ethics_checkpoint.csv",
        item_runner=run_single_item_ethics,
        result_builder=build_ethics_result,
        error_extra_fields_fn=lambda row: {'subscale': row.get('subscale', '')},
        include_confidence=include_confidence
    )


def run_moralchoice_experiment(sample_size=None, include_confidence=True):
    """Run full MoralChoice experiment (sync)."""
    mc = load_moralchoice_data(sample_size)
    if mc is None:
        return None

    return _run_experiment_sync(
        benchmark_name="MoralChoice",
        data=mc,
        checkpoint_path="results/raw/moralchoice_checkpoint.csv",
        item_runner=run_single_item_moralchoice,
        result_builder=build_moralchoice_result,
        error_extra_fields_fn=None,
        include_confidence=include_confidence
    )


def run_morables_experiment(sample_size=None, include_confidence=True):
    """Run full MORABLES experiment (sync)."""
    morables = load_morables_data(sample_size)
    if morables is None:
        return None

    return _run_experiment_sync(
        benchmark_name="MORABLES",
        data=morables,
        checkpoint_path="results/raw/morables_checkpoint.csv",
        item_runner=run_single_item_morables,
        result_builder=build_morables_result,
        error_extra_fields_fn=None,
        include_confidence=include_confidence
    )


# =============================================================================
# ASYNC EXPERIMENT RUNNERS
# =============================================================================

def load_completed_from_checkpoint(checkpoint_path: str) -> set:
    """Load completed (item_id, level, thinking, run) tuples from checkpoint."""
    if not Path(checkpoint_path).exists():
        return set()

    df = pd.read_csv(checkpoint_path)
    completed = set()
    for _, row in df.iterrows():
        key = (row['item_id'], row['level'], row['thinking'], row['run'])
        completed.add(key)
    return completed


def load_existing_results(checkpoint_path: str) -> list:
    """Load existing results from checkpoint as list of dicts."""
    if not Path(checkpoint_path).exists():
        return []

    df = pd.read_csv(checkpoint_path)
    return df.to_dict('records')


async def _run_experiment_async(
    benchmark_name: str,
    data: pd.DataFrame,
    checkpoint_path: str,
    results_queue: asyncio.Queue,
    item_runner,
    result_builder,
    include_confidence: bool = True,
    resume: bool = False
) -> List[Dict[str, Any]]:
    """
    Generic async experiment runner with resume support.

    Args:
        benchmark_name: Name for logging and queue tagging (e.g., "ethics")
        data: DataFrame with items to process
        checkpoint_path: Path to checkpoint file
        results_queue: Async queue for checkpoint writer
        item_runner: Async function(row, level, thinking, include_confidence) -> response_data
        result_builder: Function(row, level, thinking, run, response_data, include_confidence) -> result dict
        include_confidence: Whether to include confidence scoring
        resume: Whether to resume from checkpoint

    Returns:
        List of result dictionaries
    """
    # Load existing progress if resuming
    completed = set()
    results = []
    if resume:
        completed = load_completed_from_checkpoint(checkpoint_path)
        results = load_existing_results(checkpoint_path)
        print(f"{benchmark_name.upper()}: Resuming with {len(completed)} items already completed")

    total_items = N_RUNS * len(THINKING_CONDITIONS) * len(LEVELS) * len(data)
    remaining = total_items - len(completed)
    print(f"{benchmark_name.upper()}: {len(data)} items, {remaining} API calls remaining")

    benchmark_key = benchmark_name.lower()

    with tqdm(total=total_items, initial=len(completed), desc=benchmark_name.upper(), unit="item", leave=True) as pbar:
        for run in range(N_RUNS):
            for thinking in THINKING_CONDITIONS:
                for level in LEVELS:
                    thinking_label = "ON" if thinking else "OFF"
                    pbar.set_postfix(level=level, thinking=thinking_label, run=run+1)

                    for idx, row in data.iterrows():
                        # Skip if already completed
                        key = (row['item_id'], level, thinking, run)
                        if key in completed:
                            continue

                        try:
                            response_data = await item_runner(row, level, thinking, include_confidence)
                            result = result_builder(row, level, thinking, run, response_data, include_confidence)
                            result['benchmark'] = benchmark_key
                            results.append(result)
                            await results_queue.put((benchmark_key, result))
                        except Exception as e:
                            print(f"\n{benchmark_name.upper()} error on {row['item_id']}: {e}")
                            error_result = _build_error_result(row, level, thinking, run, e)
                            error_result['benchmark'] = benchmark_key
                            results.append(error_result)
                        pbar.update(1)

    print(f"{benchmark_name.upper()}: Complete with {len(results)} results")
    return results


async def run_ethics_experiment_async(results_queue: asyncio.Queue, sample_size=None, include_confidence=True, resume=False):
    """Run ETHICS experiment asynchronously."""
    ethics = load_ethics_data(sample_size)
    if ethics is None:
        return []

    return await _run_experiment_async(
        benchmark_name="ETHICS",
        data=ethics,
        checkpoint_path="results/raw/ethics_checkpoint.csv",
        results_queue=results_queue,
        item_runner=run_single_item_ethics_async,
        result_builder=build_ethics_result,
        include_confidence=include_confidence,
        resume=resume
    )


async def run_moralchoice_experiment_async(results_queue: asyncio.Queue, sample_size=None, include_confidence=True, resume=False):
    """Run MoralChoice experiment asynchronously."""
    mc = load_moralchoice_data(sample_size)
    if mc is None:
        return []

    return await _run_experiment_async(
        benchmark_name="MoralChoice",
        data=mc,
        checkpoint_path="results/raw/moralchoice_checkpoint.csv",
        results_queue=results_queue,
        item_runner=run_single_item_moralchoice_async,
        result_builder=build_moralchoice_result,
        include_confidence=include_confidence,
        resume=resume
    )


async def run_morables_experiment_async(results_queue: asyncio.Queue, sample_size=None, include_confidence=True, resume=False):
    """Run MORABLES experiment asynchronously."""
    morables = load_morables_data(sample_size)
    if morables is None:
        return []

    return await _run_experiment_async(
        benchmark_name="MORABLES",
        data=morables,
        checkpoint_path="results/raw/morables_checkpoint.csv",
        results_queue=results_queue,
        item_runner=run_single_item_morables_async,
        result_builder=build_morables_result,
        include_confidence=include_confidence,
        resume=resume
    )


async def checkpoint_writer(results_queue: asyncio.Queue, checkpoint_interval: int = 50, resume: bool = False):
    """Background task to write checkpoints as results come in."""
    # Load existing results if resuming
    all_results = {'ethics': [], 'moralchoice': [], 'morables': []}
    if resume:
        for bm in all_results.keys():
            all_results[bm] = load_existing_results(f"results/raw/{bm}_checkpoint.csv")
        total_existing = sum(len(r) for r in all_results.values())
        print(f"  [Checkpoint] Loaded {total_existing} existing results")
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
    print(f"Resume mode: {args.resume}")
    print()

    # Create output directories
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/processed").mkdir(parents=True, exist_ok=True)

    # Reset rate limiter (unless resuming, to preserve stats)
    if not args.resume:
        reset_rate_limiter()

    # Results queue for checkpoint writing
    results_queue = asyncio.Queue()

    # Start checkpoint writer
    checkpoint_task = asyncio.create_task(checkpoint_writer(results_queue, resume=args.resume))

    # Build list of experiment tasks
    tasks = []
    if args.ethics:
        tasks.append(run_ethics_experiment_async(results_queue, args.sample_size, args.confidence, args.resume))
    if args.moralchoice:
        tasks.append(run_moralchoice_experiment_async(results_queue, args.sample_size, args.confidence, args.resume))
    if args.morables:
        tasks.append(run_morables_experiment_async(results_queue, args.sample_size, args.confidence, args.resume))

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
  python run_experiment.py --async --resume   # Resume from checkpoints
        """
    )

    parser.add_argument('--async', dest='use_async', action='store_true',
                        help='Use async mode for parallel dataset execution (~2.5x faster)')
    parser.add_argument('--sample', dest='sample_size', type=int, default=SAMPLE_SIZE,
                        help=f'Sample size per benchmark (default: {SAMPLE_SIZE}, use 0 for full)')
    parser.add_argument('--no-confidence', dest='confidence', action='store_false',
                        help='Disable confidence scoring')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoints (async mode only)')

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
