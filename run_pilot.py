"""Run pilot experiment on small subset."""

import pandas as pd
import os
import re
from pathlib import Path
from tqdm import tqdm

# Load environment variables from .env
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            match = re.match(r'(\w+)\s*=\s*["\']?([^"\']+)["\']?', line)
            if match:
                os.environ[match.group(1)] = match.group(2)

from prompts import get_ethics_prompt, get_moralchoice_prompt
from src.api import call_with_rate_limit
from src.extraction import extract_ethics_answer, extract_moralchoice_answer
import config

# Pilot settings
PILOT_N = 5  # Items per benchmark
LEVELS = [0, 2, 4]  # Subset of levels to test
THINKING_CONDITIONS = [False, True]


def run_ethics_pilot():
    """Run pilot on ETHICS subset."""

    ethics = pd.read_csv("data/ethics_sample.csv").head(PILOT_N)
    results = []

    for thinking in THINKING_CONDITIONS:
        thinking_label = "ON" if thinking else "OFF"

        for level in LEVELS:
            print(f"\nRunning ETHICS Level {level}, Thinking {thinking_label}")

            for _, row in tqdm(ethics.iterrows(), total=len(ethics), desc=f"L{level}"):

                # Handle Level 5 (two-pass) separately
                if level == 5:
                    # Pass 1
                    prompt1 = get_ethics_prompt(5, row['scenario'])
                    response1 = call_with_rate_limit(prompt1, thinking)

                    # Pass 2
                    prompt2 = get_ethics_prompt(5, row['scenario'], response1.content)
                    response2 = call_with_rate_limit(prompt2, thinking)

                    content = response2.content
                    full_response = f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}"
                    thinking_content = response2.thinking
                else:
                    prompt = get_ethics_prompt(level, row['scenario'])
                    response = call_with_rate_limit(prompt, thinking)
                    content = response.content
                    full_response = content
                    thinking_content = response.thinking

                # Extract answer
                extracted = extract_ethics_answer(content)

                results.append({
                    'item_id': row['item_id'],
                    'subscale': row['subscale'],
                    'level': level,
                    'thinking': thinking,
                    'response': full_response,
                    'thinking_content': thinking_content,
                    'extracted_answer': extracted,
                    'correct_answer': row['label'],
                    'correct': extracted == row['label'] if extracted else None
                })

    return pd.DataFrame(results)


def run_moralchoice_pilot():
    """Run pilot on MoralChoice subset."""

    mc = pd.read_csv("data/moralchoice_sample.csv").head(PILOT_N)
    results = []

    for thinking in THINKING_CONDITIONS:
        thinking_label = "ON" if thinking else "OFF"

        for level in LEVELS:
            print(f"\nRunning MoralChoice Level {level}, Thinking {thinking_label}")

            for _, row in tqdm(mc.iterrows(), total=len(mc), desc=f"L{level}"):

                if level == 5:
                    prompt1 = get_moralchoice_prompt(5, row['context'], row['option_a'], row['option_b'])
                    response1 = call_with_rate_limit(prompt1, thinking)

                    prompt2 = get_moralchoice_prompt(5, row['context'], row['option_a'], row['option_b'],
                                                     response1.content)
                    response2 = call_with_rate_limit(prompt2, thinking)

                    content = response2.content
                    full_response = f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}"
                    thinking_content = response2.thinking
                else:
                    prompt = get_moralchoice_prompt(level, row['context'], row['option_a'], row['option_b'])
                    response = call_with_rate_limit(prompt, thinking)
                    content = response.content
                    full_response = content
                    thinking_content = response.thinking

                extracted = extract_moralchoice_answer(content)

                results.append({
                    'item_id': row['item_id'],
                    'level': level,
                    'thinking': thinking,
                    'response': full_response,
                    'thinking_content': thinking_content,
                    'extracted_answer': extracted,
                })

    return pd.DataFrame(results)


def analyze_pilot(ethics_results, mc_results):
    """Quick analysis of pilot results."""

    print("\n" + "=" * 60)
    print("PILOT RESULTS")
    print("=" * 60)

    # ETHICS
    print("\nETHICS Accuracy by Condition:")
    ethics_summary = ethics_results.groupby(['level', 'thinking']).agg({
        'correct': 'mean',
        'extracted_answer': lambda x: x.isna().mean()  # Extraction failure rate
    }).round(3)
    ethics_summary.columns = ['accuracy', 'extraction_failure']
    print(ethics_summary)

    # MoralChoice
    print("\nMoralChoice Extraction Success:")
    mc_summary = mc_results.groupby(['level', 'thinking']).agg({
        'extracted_answer': lambda x: x.notna().mean()
    }).round(3)
    mc_summary.columns = ['extraction_success']
    print(mc_summary)

    # Response lengths
    print("\nResponse Lengths (mean words):")
    ethics_results['response_length'] = ethics_results['response'].str.split().str.len()
    length_summary = ethics_results.groupby(['level', 'thinking'])['response_length'].mean().round(0)
    print(length_summary)

    # Thinking content check
    print("\nThinking Content Present (when enabled):")
    thinking_on = ethics_results[ethics_results['thinking'] == True]
    has_thinking = thinking_on['thinking_content'].notna().mean()
    print(f"  ETHICS: {has_thinking:.1%}")

    thinking_on_mc = mc_results[mc_results['thinking'] == True]
    has_thinking_mc = thinking_on_mc['thinking_content'].notna().mean()
    print(f"  MoralChoice: {has_thinking_mc:.1%}")

    return ethics_summary, mc_summary


def main():
    print("=" * 60)
    print("STARTING PILOT RUN")
    print("=" * 60)
    print(f"Model: {config.MODEL}")
    print(f"Items per benchmark: {PILOT_N}")
    print(f"Levels: {LEVELS}")
    print(f"Thinking conditions: {THINKING_CONDITIONS}")
    print(f"Total API calls: {PILOT_N * len(LEVELS) * len(THINKING_CONDITIONS) * 2}")

    # Create output directory
    Path("results/pilot").mkdir(parents=True, exist_ok=True)

    # Run pilots
    print("\n--- Running ETHICS Pilot ---")
    ethics_results = run_ethics_pilot()

    print("\n--- Running MoralChoice Pilot ---")
    mc_results = run_moralchoice_pilot()

    # Save results
    ethics_results.to_csv("results/pilot/ethics_pilot.csv", index=False)
    mc_results.to_csv("results/pilot/moralchoice_pilot.csv", index=False)

    # Analyze
    analyze_pilot(ethics_results, mc_results)

    print("\n" + "=" * 60)
    print("PILOT COMPLETE")
    print("=" * 60)
    print("Results saved to:")
    print("  - results/pilot/ethics_pilot.csv")
    print("  - results/pilot/moralchoice_pilot.csv")


if __name__ == "__main__":
    main()
