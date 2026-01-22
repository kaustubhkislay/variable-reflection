"""Prepare MORABLES dataset for experiment."""

import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
import config

np.random.seed(config.RANDOM_SEED)


def load_morables_data():
    """Load MORABLES dataset from HuggingFace."""

    print("Loading MORABLES dataset from HuggingFace...")
    print("  Source: cardiffnlp/Morables")

    # Load dataset
    dataset = load_dataset("cardiffnlp/Morables")

    # Get the test split (or train if test not available)
    if 'test' in dataset:
        data = dataset['test']
    elif 'train' in dataset:
        data = dataset['train']
    else:
        # Use first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"  Using split: {split_name}")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    print(f"  Loaded {len(df)} fables")
    print(f"  Columns: {list(df.columns)}")

    return df


def prepare_morables_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare MORABLES data for experiment.

    Standardizes column names and formats options.
    """
    print("\nPreparing data...")

    # Print sample to understand structure
    print(f"  Sample columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"  Sample row keys: {df.iloc[0].to_dict().keys()}")

    # Standardize column names based on actual dataset structure
    # The HuggingFace dataset may have different column names
    column_mapping = {
        'story': 'fable',
        'text': 'fable',
        'fable_text': 'fable',
        'narrative': 'fable',
        'moral': 'correct_moral',
        'correct_moral': 'correct_moral',
        'label': 'correct_idx',
        'answer': 'correct_idx',
        'correct_answer': 'correct_idx',
    }

    # Apply mapping for columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Handle options - they might be in a list column or separate columns
    if 'options' in df.columns:
        # Options in a single list column
        df['option_a'] = df['options'].apply(lambda x: x[0] if len(x) > 0 else '')
        df['option_b'] = df['options'].apply(lambda x: x[1] if len(x) > 1 else '')
        df['option_c'] = df['options'].apply(lambda x: x[2] if len(x) > 2 else '')
        df['option_d'] = df['options'].apply(lambda x: x[3] if len(x) > 3 else '')
        df['option_e'] = df['options'].apply(lambda x: x[4] if len(x) > 4 else '')
    elif 'choices' in df.columns:
        # Choices in a single list column
        df['option_a'] = df['choices'].apply(lambda x: x[0] if len(x) > 0 else '')
        df['option_b'] = df['choices'].apply(lambda x: x[1] if len(x) > 1 else '')
        df['option_c'] = df['choices'].apply(lambda x: x[2] if len(x) > 2 else '')
        df['option_d'] = df['choices'].apply(lambda x: x[3] if len(x) > 3 else '')
        df['option_e'] = df['choices'].apply(lambda x: x[4] if len(x) > 4 else '')

    # Ensure correct_idx is integer (0-4 for A-E)
    if 'correct_idx' in df.columns:
        # Handle if it's a letter
        if df['correct_idx'].dtype == object:
            letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
                           'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
            df['correct_idx'] = df['correct_idx'].map(
                lambda x: letter_to_idx.get(x, x) if isinstance(x, str) else x
            )
        df['correct_idx'] = df['correct_idx'].astype(int)

    # Add item IDs
    df['item_id'] = [f"morables_{i}" for i in range(len(df))]

    # Select and order columns
    required_cols = ['item_id', 'fable', 'option_a', 'option_b', 'option_c',
                     'option_d', 'option_e', 'correct_idx']

    # Add optional columns if they exist
    optional_cols = ['correct_moral', 'source', 'difficulty']
    for col in optional_cols:
        if col in df.columns:
            required_cols.append(col)

    # Filter to required columns that exist
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]

    print(f"  Final columns: {df.columns.tolist()}")
    print(f"  Total items: {len(df)}")

    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Validate the prepared dataset."""

    print("\nValidating data...")

    issues = []

    # Check required columns
    required = ['item_id', 'fable', 'option_a', 'option_b', 'option_c',
                'option_d', 'option_e', 'correct_idx']
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Check for empty fables
    empty_fables = df['fable'].isna().sum() + (df['fable'] == '').sum()
    if empty_fables > 0:
        issues.append(f"Empty fables: {empty_fables}")

    # Check correct_idx range
    if 'correct_idx' in df.columns:
        invalid_idx = ((df['correct_idx'] < 0) | (df['correct_idx'] > 4)).sum()
        if invalid_idx > 0:
            issues.append(f"Invalid correct_idx values: {invalid_idx}")

    # Check for empty options
    for opt in ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']:
        if opt in df.columns:
            empty = df[opt].isna().sum() + (df[opt] == '').sum()
            if empty > 0:
                issues.append(f"Empty {opt}: {empty}")

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  All validations passed!")
        return True


def main():
    """Prepare and save MORABLES dataset."""

    print("=" * 50)
    print("PREPARING MORABLES DATASET")
    print("=" * 50)

    # Create output directory
    Path("data/morables").mkdir(parents=True, exist_ok=True)

    # Load from HuggingFace
    df = load_morables_data()

    # Prepare data
    df = prepare_morables_data(df)

    # Validate
    valid = validate_data(df)

    if not valid:
        print("\nWARNING: Data validation failed. Please check the dataset structure.")
        print("You may need to adjust the column mapping in prepare_morables_data().")

    # Save
    output_path = "data/morables/morables_full.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved full dataset to: {output_path}")

    # Create a sample for pilot testing
    sample_size = min(100, len(df))
    sample = df.sample(n=sample_size, random_state=config.RANDOM_SEED)
    sample_path = "data/morables/morables_sample.csv"
    sample.to_csv(sample_path, index=False)
    print(f"Saved sample ({sample_size} items) to: {sample_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("MORABLES PREPARATION COMPLETE")
    print("=" * 50)
    print(f"  Full dataset: {output_path} ({len(df)} items)")
    print(f"  Sample: {sample_path} ({sample_size} items)")

    # Show sample item
    print("\n--- Sample Item ---")
    sample_item = df.iloc[0]
    print(f"Fable: {sample_item['fable'][:200]}...")
    print(f"Options:")
    print(f"  A) {sample_item['option_a'][:80]}...")
    print(f"  B) {sample_item['option_b'][:80]}...")
    print(f"  C) {sample_item['option_c'][:80]}...")
    print(f"  D) {sample_item['option_d'][:80]}...")
    print(f"  E) {sample_item['option_e'][:80]}...")
    print(f"Correct: {['A', 'B', 'C', 'D', 'E'][sample_item['correct_idx']]}")


if __name__ == "__main__":
    main()
