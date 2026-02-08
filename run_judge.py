"""Run LLM-as-Judge flip-flop detection on existing experiment results."""

import argparse
import asyncio
import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any

# Load environment variables from .env
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            match = re.match(r'(\w+)\s*=\s*["\']?([^"\']+)["\']?', line)
            if match:
                os.environ[match.group(1)] = match.group(2)

import config
from src.gemini_api import is_gemini_available, reset_gemini_rate_limiter
from src.judge import should_judge_row, judge_single_row_async


# Source file paths
SOURCE_FILES = {
    'claude': {
        'ethics': 'results/processed/ethics_results.csv',
        'moralchoice': 'results/processed/moralchoice_results.csv',
        'morables': 'results/processed/morables_results.csv',
    },
    'gemini': {
        'ethics': 'results/processed/gemini_ethics_results.csv',
        'moralchoice': 'results/processed/gemini_moralchoice_results.csv',
        'morables': 'results/processed/gemini_morables_results.csv',
    }
}


def _checkpoint_key(row: dict, source: str) -> tuple:
    """Build a unique key for resume support."""
    item_id = row['item_id']
    run = row['run']
    if source == 'claude':
        return (item_id, str(row.get('level', '')), str(row.get('thinking', '')), str(run))
    else:
        return (item_id, str(row.get('thinking_level', '')), str(row.get('prompt_level', '')), str(run))


def _checkpoint_path(source: str, benchmark: str) -> str:
    """Get checkpoint file path."""
    if source == 'claude':
        return f"results/raw/judge_{benchmark}_checkpoint.csv"
    return f"results/raw/judge_gemini_{benchmark}_checkpoint.csv"


def _output_path(source: str, benchmark: str) -> str:
    """Get final output file path."""
    if source == 'claude':
        return f"results/processed/judge_{benchmark}_results.csv"
    return f"results/processed/judge_gemini_{benchmark}_results.csv"


def load_completed_keys(checkpoint_path: str, source: str) -> set:
    """Load completed keys from checkpoint for resume support."""
    if not Path(checkpoint_path).exists():
        return set()
    df = pd.read_csv(checkpoint_path)
    completed = set()
    for _, row in df.iterrows():
        completed.add(_checkpoint_key(row.to_dict(), source))
    return completed


def load_existing_results(checkpoint_path: str) -> list:
    """Load existing results from checkpoint as list of dicts."""
    if not Path(checkpoint_path).exists():
        return []
    return pd.read_csv(checkpoint_path).to_dict('records')


def _build_output_row(row: dict, source: str, benchmark: str, judge_result: dict) -> dict:
    """Build the output row combining source info with judge results."""
    output = {
        'item_id': row['item_id'],
        'benchmark': benchmark,
        'source_model': source,
    }

    # Add condition columns (different for Claude vs Gemini)
    if source == 'claude':
        output['level'] = row.get('level')
        output['thinking'] = row.get('thinking')
    else:
        output['prompt_level'] = row.get('prompt_level')
        output['thinking_level'] = row.get('thinking_level')

    output['run'] = row['run']
    output['extracted_answer'] = row.get('extracted_answer')

    # Add ground truth
    if 'correct' in row:
        output['correct'] = row['correct']
    if 'label' in row:
        output['label'] = row['label']
    if 'correct_answer' in row:
        output['correct_answer'] = row['correct_answer']

    # Add judge results
    output.update(judge_result)
    output['timestamp'] = datetime.now().isoformat()

    return output


async def process_benchmark_async(
    df: pd.DataFrame,
    benchmark: str,
    source: str,
    resume: bool = False
) -> List[Dict[str, Any]]:
    """Process one benchmark's results through the judge."""
    cp_path = _checkpoint_path(source, benchmark)

    # Load completed items if resuming
    completed_keys = set()
    results = []
    if resume:
        completed_keys = load_completed_keys(cp_path, source)
        results = load_existing_results(cp_path)
        print(f"  Resuming with {len(completed_keys)} already judged")

    # Filter to rows worth judging
    judgeable_mask = df.apply(lambda r: should_judge_row(r.to_dict()), axis=1)
    judgeable = df[judgeable_mask]
    skipped = len(df) - len(judgeable)

    print(f"  {len(judgeable)} judgeable rows ({skipped} skipped as too terse)")

    checkpoint_interval = 50
    count = 0

    with tqdm(total=len(judgeable), initial=len(completed_keys),
              desc=f"{source}/{benchmark}", unit="item") as pbar:
        for _, row in judgeable.iterrows():
            row_dict = row.to_dict()
            key = _checkpoint_key(row_dict, source)

            if key in completed_keys:
                continue

            try:
                judge_result = await judge_single_row_async(row_dict, benchmark)
                output_row = _build_output_row(row_dict, source, benchmark, judge_result)
                results.append(output_row)
            except Exception as e:
                print(f"\n  Error judging {row_dict['item_id']}: {e}")
                results.append({
                    'item_id': row_dict['item_id'],
                    'benchmark': benchmark,
                    'source_model': source,
                    'judge_error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })

            count += 1
            pbar.update(1)

            # Checkpoint periodically
            if count % checkpoint_interval == 0:
                pd.DataFrame(results).to_csv(cp_path, index=False)

    # Final checkpoint
    if results:
        pd.DataFrame(results).to_csv(cp_path, index=False)

    return results


async def run_judge_async(args):
    """Run the judge pipeline."""
    start_time = datetime.now()

    print("=" * 60)
    print("LLM-AS-JUDGE: FLIP-FLOP DETECTION")
    print("=" * 60)
    print(f"Start time: {start_time}")
    print(f"Judge model: Gemini 3 Flash")
    print(f"Thinking level: {config.JUDGE_THINKING_LEVEL}")
    print(f"Max tokens: {config.JUDGE_MAX_TOKENS}")
    print(f"Min trace words: {config.JUDGE_MIN_TRACE_WORDS}")
    print(f"Resume: {args.resume}")
    print()

    # Create output directories
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/processed").mkdir(parents=True, exist_ok=True)

    if not args.resume:
        reset_gemini_rate_limiter()

    # Determine which sources and benchmarks to process
    sources = []
    if args.claude:
        sources.append('claude')
    if args.gemini:
        sources.append('gemini')
    if not sources:
        sources = ['claude', 'gemini']

    benchmarks = []
    if args.ethics:
        benchmarks.append('ethics')
    if args.moralchoice:
        benchmarks.append('moralchoice')
    if args.morables:
        benchmarks.append('morables')
    if not benchmarks:
        benchmarks = ['ethics', 'moralchoice', 'morables']

    # Process each source/benchmark combination
    all_results = {}

    for source in sources:
        for benchmark in benchmarks:
            csv_path = SOURCE_FILES[source][benchmark]
            if not Path(csv_path).exists():
                print(f"\nSkipping {source}/{benchmark}: {csv_path} not found")
                continue

            print(f"\n{'=' * 60}")
            print(f"JUDGING: {source.upper()} / {benchmark.upper()}")
            print(f"{'=' * 60}")

            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} rows from {csv_path}")

            if args.dry_run:
                judgeable = df.apply(lambda r: should_judge_row(r.to_dict()), axis=1).sum()
                print(f"  Would judge {judgeable} rows (dry run)")
                continue

            results = await process_benchmark_async(
                df, benchmark, source, resume=args.resume
            )

            # Save final results
            if results:
                out_path = _output_path(source, benchmark)
                pd.DataFrame(results).to_csv(out_path, index=False)
                print(f"  Saved {len(results)} judge results to {out_path}")

            key = f"{source}_{benchmark}"
            all_results[key] = results

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'=' * 60}")
    print(f"JUDGE PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Duration: {elapsed}")
    for key, results in all_results.items():
        if results:
            n_flips = sum(1 for r in results if r.get('flip_flop_detected') == True)
            print(f"  {key}: {len(results)} judged, {n_flips} flip-flops detected")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-Judge flip-flop detection on experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_judge.py                    # Judge all benchmarks (Claude + Gemini)
  python run_judge.py --claude           # Judge only Claude results
  python run_judge.py --gemini           # Judge only Gemini results
  python run_judge.py --ethics           # Judge only ethics benchmark
  python run_judge.py --resume           # Resume from checkpoint
  python run_judge.py --dry-run          # Count rows, estimate cost
        """
    )

    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoints')
    parser.add_argument('--dry-run', action='store_true',
                        help='Count judgeable rows without calling API')

    # Source selection
    parser.add_argument('--claude', action='store_true',
                        help='Judge only Claude experiment results')
    parser.add_argument('--gemini', action='store_true',
                        help='Judge only Gemini experiment results')

    # Benchmark selection
    parser.add_argument('--ethics', action='store_true',
                        help='Judge only ETHICS benchmark')
    parser.add_argument('--moralchoice', action='store_true',
                        help='Judge only MoralChoice benchmark')
    parser.add_argument('--morables', action='store_true',
                        help='Judge only MORABLES benchmark')

    args = parser.parse_args()

    if not is_gemini_available():
        print("ERROR: Gemini via OpenRouter not available.")
        print("  1. Ensure openai package is installed")
        print("  2. Set OPENROUTER_API_KEY in your .env file")
        return

    asyncio.run(run_judge_async(args))


if __name__ == "__main__":
    main()
