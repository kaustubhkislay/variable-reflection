"""Run MORABLES experiment."""

import pandas as pd
import os
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Load environment variables from .env
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            match = re.match(r'(\w+)\s*=\s*["\']?([^"\']+)["\']?', line)
            if match:
                os.environ[match.group(1)] = match.group(2)

from prompts import get_morables_prompt
from src.api import call_with_rate_limit, APIResponse
from src.extraction import (
    extract_morables_answer,
    extract_confidence_score,
    count_reasoning_markers,
    count_uncertainty_markers,
    categorize_confidence
)
import config

# Experiment settings
LEVELS = [0, 1, 2, 3, 4, 5]
THINKING_CONDITIONS = [False, True]
N_RUNS = config.N_RUNS

# Sample size (set to None for full dataset)
SAMPLE_SIZE = 100


def run_single_item_morables(row, level, thinking, include_confidence=True):
    """Run single MORABLES item at given condition."""

    # Get options as list
    options = [
        row['option_a'],
        row['option_b'],
        row['option_c'],
        row['option_d'],
        row['option_e']
    ]

    if level == 5:
        # Two-pass
        prompt1 = get_morables_prompt(5, row['fable'], options,
                                       include_confidence=include_confidence)
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
        prompt = get_morables_prompt(level, row['fable'], options,
                                      include_confidence=include_confidence)
        response = call_with_rate_limit(prompt, thinking)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


def run_morables_experiment(include_confidence=True):
    """Run full MORABLES experiment."""

    # Load data
    data_path = "data/morables/morables_sample.csv"
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found.")
        print("Please run 'python prepare_morables_data.py' first.")
        return None

    morables = pd.read_csv(data_path)

    if SAMPLE_SIZE and len(morables) > SAMPLE_SIZE:
        morables = morables.sample(n=SAMPLE_SIZE, random_state=config.RANDOM_SEED)
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
                        response_data = run_single_item_morables(
                            row, level, thinking, include_confidence
                        )

                        extracted = extract_morables_answer(response_data['content'])
                        confidence = extract_confidence_score(response_data['content']) if include_confidence else None

                        # Map extracted letter to index for correctness check
                        letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                        extracted_idx = letter_to_idx.get(extracted)
                        correct = extracted_idx == row['correct_idx'] if extracted_idx is not None else None

                        results.append({
                            'item_id': row['item_id'],
                            'fable': row['fable'][:200],  # Truncate for storage
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

                # Save checkpoint after each condition
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv("results/raw/morables_checkpoint.csv", index=False)

    return pd.DataFrame(results)


def main():
    start_time = datetime.now()
    print("=" * 60)
    print("STARTING MORABLES EXPERIMENT")
    print("=" * 60)
    print(f"Start time: {start_time}")
    print(f"Model: {config.MODEL}")
    print(f"Levels: {LEVELS}")
    print(f"Thinking conditions: {THINKING_CONDITIONS}")
    print(f"Runs: {N_RUNS}")

    # Create output directories
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/processed").mkdir(parents=True, exist_ok=True)

    # Calculate estimated API calls
    data_path = "data/morables/morables_sample.csv"
    if Path(data_path).exists():
        morables_items = min(SAMPLE_SIZE, len(pd.read_csv(data_path))) if SAMPLE_SIZE else len(pd.read_csv(data_path))
    else:
        print(f"\nError: {data_path} not found.")
        print("Please run 'python prepare_morables_data.py' first.")
        return

    # Level 5 requires 2 calls per item
    morables_calls = morables_items * (5 * 1 + 1 * 2) * 2 * N_RUNS

    print(f"\nDataset size: {morables_items} items")
    print(f"Estimated API calls: ~{morables_calls}")

    # Run experiment
    print("\n" + "=" * 60)
    print("RUNNING MORABLES EXPERIMENT")
    print("=" * 60)
    morables_results = run_morables_experiment(include_confidence=True)

    if morables_results is not None:
        morables_results.to_csv("results/processed/morables_results.csv", index=False)
        print(f"\nMORABLES complete: {len(morables_results)} observations")

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Results saved to: results/processed/morables_results.csv")

        # Print quick summary
        if len(morables_results) > 0:
            valid_results = morables_results[morables_results['correct'].notna()]
            if len(valid_results) > 0:
                print(f"\nQuick Summary:")
                print(f"  Overall accuracy: {valid_results['correct'].mean():.3f}")
                print(f"  Extraction success rate: {morables_results['extracted_answer'].notna().mean():.3f}")


if __name__ == "__main__":
    main()
