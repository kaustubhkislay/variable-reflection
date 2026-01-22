"""Run full analysis pipeline."""

import pandas as pd
from pathlib import Path
from src.analysis import run_all_analyses, compute_moralchoice_consistency
from src.visualize import create_all_figures


def load_results():
    """Load experiment results, falling back to checkpoints if needed."""

    # Try processed results first, then checkpoints
    try:
        ethics = pd.read_csv("results/processed/ethics_results.csv")
        print(f"Loaded ETHICS from processed: {len(ethics)} observations")
    except FileNotFoundError:
        ethics = pd.read_csv("results/raw/ethics_checkpoint.csv")
        print(f"Loaded ETHICS from checkpoint: {len(ethics)} observations")

    try:
        mc = pd.read_csv("results/processed/moralchoice_results.csv")
        print(f"Loaded MoralChoice from processed: {len(mc)} observations")
    except FileNotFoundError:
        mc = pd.read_csv("results/raw/moralchoice_checkpoint.csv")
        print(f"Loaded MoralChoice from checkpoint: {len(mc)} observations")

    return ethics, mc


def print_summary_stats(ethics, mc):
    """Print basic summary statistics."""

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print("\nETHICS:")
    print(f"  Total observations: {len(ethics)}")
    print(f"  Unique items: {ethics['item_id'].nunique()}")
    print(f"  Levels: {sorted(ethics['level'].unique())}")
    print(f"  Runs: {sorted(ethics['run'].unique())}")
    print(f"  Extraction failures: {ethics['extracted_answer'].isna().sum()} ({ethics['extracted_answer'].isna().mean()*100:.1f}%)")
    if 'error' in ethics.columns:
        print(f"  API errors: {ethics['error'].notna().sum()}")

    print("\nMoralChoice:")
    print(f"  Total observations: {len(mc)}")
    print(f"  Unique items: {mc['item_id'].nunique()}")
    print(f"  Levels: {sorted(mc['level'].unique())}")
    print(f"  Runs: {sorted(mc['run'].unique())}")
    print(f"  Extraction failures: {mc['extracted_answer'].isna().sum()} ({mc['extracted_answer'].isna().mean()*100:.1f}%)")
    if 'ambiguity' in mc.columns:
        print(f"  Ambiguity distribution: {mc.groupby('ambiguity')['item_id'].nunique().to_dict()}")


def save_analysis_tables(analysis_results, output_dir='outputs/tables'):
    """Save analysis results to CSV files."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nSaving tables to {output_dir}/...")

    for name, result in analysis_results.items():
        if result is None:
            continue
        if isinstance(result, pd.DataFrame):
            filepath = f"{output_dir}/{name}.csv"
            result.to_csv(filepath)
            print(f"  - {name}.csv")
        elif isinstance(result, dict) and name not in ['trends', 'mc_kruskal']:
            # Convert dict results to DataFrame if possible
            try:
                df = pd.DataFrame(result)
                filepath = f"{output_dir}/{name}.csv"
                df.to_csv(filepath)
                print(f"  - {name}.csv")
            except:
                pass


def calculate_cost(ethics, mc):
    """Calculate estimated API cost."""

    input_tokens = ethics['input_tokens'].sum() + mc['input_tokens'].sum()
    output_tokens = ethics['output_tokens'].sum() + mc['output_tokens'].sum()

    # Claude Haiku 4.5 pricing
    input_cost = (input_tokens / 1_000_000) * 1.00
    output_cost = (output_tokens / 1_000_000) * 5.00
    total_cost = input_cost + output_cost

    print("\n" + "=" * 60)
    print("TOKEN USAGE & COST")
    print("=" * 60)
    print(f"  Input tokens:  {input_tokens:,}")
    print(f"  Output tokens: {output_tokens:,}")
    print(f"  Input cost:    ${input_cost:.2f}")
    print(f"  Output cost:   ${output_cost:.2f}")
    print(f"  Total cost:    ${total_cost:.2f}")


def main():
    print("=" * 60)
    print("REFLECTION STUDY ANALYSIS")
    print("=" * 60)

    # Load results
    print("\nLoading results...")
    ethics, mc = load_results()

    # Print summary
    print_summary_stats(ethics, mc)

    # Calculate cost
    calculate_cost(ethics, mc)

    # Run analyses
    print("\n")
    analysis_results = run_all_analyses(ethics, mc)

    # Compute consistency for visualization
    consistency_df, _ = compute_moralchoice_consistency(mc)

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    create_all_figures(ethics, mc, consistency_df)

    # Save tables
    print("\n" + "=" * 60)
    print("SAVING TABLES")
    print("=" * 60)
    save_analysis_tables(analysis_results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  - Figures: outputs/figures/")
    print("  - Tables:  outputs/tables/")


if __name__ == "__main__":
    main()
