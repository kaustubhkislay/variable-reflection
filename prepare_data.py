"""Prepare experimental data samples for all benchmarks."""

import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
import config

np.random.seed(config.RANDOM_SEED)


def load_ethics_data():
    """Load and sample ETHICS benchmark data."""

    data_frames = []

    # Commonsense - direct wrong/not wrong judgments
    print("Loading commonsense data...")
    cm = pd.read_csv("data/ethics/commonsense.csv")
    cm['subscale'] = 'commonsense'
    cm['scenario'] = cm['input']  # Rename column
    cm['label'] = cm['label'].map({0: 'not wrong', 1: 'wrong'})
    cm = cm[['scenario', 'label', 'subscale']]
    data_frames.append(cm)
    print(f"  Loaded {len(cm)} commonsense items")

    # Deontology - combine scenario and excuse
    print("Loading deontology data...")
    deont = pd.read_csv("data/ethics/deontology.csv")
    deont['subscale'] = 'deontology'
    # Format: "Scenario: X. Excuse: Y" - is the excuse reasonable?
    deont['scenario'] = deont.apply(
        lambda row: f"{row['scenario']} Response: \"{row['excuse']}\"",
        axis=1
    )
    # 0 = unreasonable (wrong to use this excuse), 1 = reasonable (not wrong)
    deont['label'] = deont['label'].map({0: 'wrong', 1: 'not wrong'})
    deont = deont[['scenario', 'label', 'subscale']]
    data_frames.append(deont)
    print(f"  Loaded {len(deont)} deontology items")

    # Virtue - parse [SEP] format
    print("Loading virtue data...")
    virtue = pd.read_csv("data/ethics/virtue.csv")
    virtue['subscale'] = 'virtue'
    # Format: "scenario [SEP] trait" - is the trait appropriate?
    def parse_virtue(row):
        parts = row['scenario'].split(' [SEP] ')
        if len(parts) == 2:
            return f"{parts[0]} The person is described as: {parts[1]}."
        return row['scenario']
    virtue['scenario'] = virtue.apply(parse_virtue, axis=1)
    # 1 = appropriate trait (not wrong), 0 = inappropriate trait (wrong)
    virtue['label'] = virtue['label'].map({0: 'wrong', 1: 'not wrong'})
    virtue = virtue[['scenario', 'label', 'subscale']]
    data_frames.append(virtue)
    print(f"  Loaded {len(virtue)} virtue items")

    # Combine all
    ethics = pd.concat(data_frames, ignore_index=True)

    # Sample 500 per subscale (or all if fewer)
    print("\nSampling data...")
    sampled_list = []
    for subscale in ethics['subscale'].unique():
        subset = ethics[ethics['subscale'] == subscale]
        n_sample = min(500, len(subset))
        sampled_list.append(subset.sample(n=n_sample, random_state=config.RANDOM_SEED))
    sampled = pd.concat(sampled_list, ignore_index=True)

    # Add item IDs
    sampled['item_id'] = [f"ethics_{i}" for i in range(len(sampled))]

    return sampled


def load_moralchoice_data():
    """Load MoralChoice dilemmas."""

    print("Loading MoralChoice data...")
    mc = pd.read_csv("data/moralchoice/dilemmas.csv")

    # Ensure required columns exist
    required = ['context', 'option_a', 'option_b']
    for col in required:
        assert col in mc.columns, f"Missing column: {col}"

    # Sample if needed
    if len(mc) > 500:
        mc = mc.sample(n=500, random_state=config.RANDOM_SEED)
        print(f"  Sampled 500 items from {len(mc)} total")
    else:
        print(f"  Loaded {len(mc)} items")

    # Add item IDs
    mc['item_id'] = [f"mc_{i}" for i in range(len(mc))]

    return mc


def load_morables_data():
    """Load MORABLES dataset from HuggingFace."""

    print("Loading MORABLES dataset from HuggingFace...")
    print("  Source: cardiffnlp/Morables (mcqa config)")

    # Load dataset with mcqa config (5-way multiple choice)
    dataset = load_dataset("cardiffnlp/Morables", "mcqa")

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
    print("\nPreparing MORABLES data...")

    # Standardize column names based on actual dataset structure
    # cardiffnlp/Morables mcqa config uses:
    #   story -> fable, choices -> options, correct_moral_label -> correct_idx
    column_mapping = {
        'story': 'fable',
        'text': 'fable',
        'fable_text': 'fable',
        'narrative': 'fable',
        'moral': 'correct_moral',
        'correct_moral_label': 'correct_idx',
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
        df['option_a'] = df['options'].apply(lambda x: x[0] if len(x) > 0 else '')
        df['option_b'] = df['options'].apply(lambda x: x[1] if len(x) > 1 else '')
        df['option_c'] = df['options'].apply(lambda x: x[2] if len(x) > 2 else '')
        df['option_d'] = df['options'].apply(lambda x: x[3] if len(x) > 3 else '')
        df['option_e'] = df['options'].apply(lambda x: x[4] if len(x) > 4 else '')
    elif 'choices' in df.columns:
        df['option_a'] = df['choices'].apply(lambda x: x[0] if len(x) > 0 else '')
        df['option_b'] = df['choices'].apply(lambda x: x[1] if len(x) > 1 else '')
        df['option_c'] = df['choices'].apply(lambda x: x[2] if len(x) > 2 else '')
        df['option_d'] = df['choices'].apply(lambda x: x[3] if len(x) > 3 else '')
        df['option_e'] = df['choices'].apply(lambda x: x[4] if len(x) > 4 else '')

    # Ensure correct_idx is integer (0-4 for A-E)
    if 'correct_idx' in df.columns:
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


def validate_morables_data(df: pd.DataFrame) -> bool:
    """Validate the prepared MORABLES dataset."""

    print("\nValidating MORABLES data...")

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
    """Prepare and save experimental datasets."""

    print("=" * 50)
    print("PREPARING EXPERIMENTAL DATA")
    print("=" * 50)

    # Load and prepare ETHICS
    print("\n--- ETHICS Benchmark ---")
    ethics = load_ethics_data()
    print(f"\nTotal sampled: {len(ethics)} items")
    print(f"Subscales: {ethics['subscale'].value_counts().to_dict()}")
    print(f"Labels: {ethics['label'].value_counts().to_dict()}")

    # Save
    ethics.to_csv("data/ethics_sample.csv", index=False)
    print(f"Saved to: data/ethics_sample.csv")

    # Load and prepare MoralChoice
    print("\n--- MoralChoice Dataset ---")
    mc = load_moralchoice_data()
    print(f"Total: {len(mc)} items")
    if 'ambiguity' in mc.columns:
        print(f"Ambiguity: {mc['ambiguity'].value_counts().to_dict()}")

    # Save
    mc.to_csv("data/moralchoice_sample.csv", index=False)
    print(f"Saved to: data/moralchoice_sample.csv")

    # Load and prepare MORABLES
    print("\n--- MORABLES Benchmark ---")
    Path("data/morables").mkdir(parents=True, exist_ok=True)

    morables_raw = load_morables_data()
    morables = prepare_morables_data(morables_raw)
    valid = validate_morables_data(morables)

    if not valid:
        print("\nWARNING: MORABLES data validation failed. Check dataset structure.")

    # Save full dataset
    morables.to_csv("data/morables/morables_full.csv", index=False)
    print(f"Saved full dataset to: data/morables/morables_full.csv")

    # Create a sample for pilot testing
    sample_size = min(100, len(morables))
    morables_sample = morables.sample(n=sample_size, random_state=config.RANDOM_SEED)
    morables_sample.to_csv("data/morables/morables_sample.csv", index=False)
    print(f"Saved sample ({sample_size} items) to: data/morables/morables_sample.csv")

    # Show sample item
    print("\n--- Sample MORABLES Item ---")
    sample_item = morables.iloc[0]
    print(f"Fable: {sample_item['fable'][:200]}...")
    print(f"Options:")
    print(f"  A) {sample_item['option_a'][:80]}...")
    print(f"  B) {sample_item['option_b'][:80]}...")
    print(f"  C) {sample_item['option_c'][:80]}...")
    print(f"  D) {sample_item['option_d'][:80]}...")
    print(f"  E) {sample_item['option_e'][:80]}...")
    print(f"Correct: {['A', 'B', 'C', 'D', 'E'][sample_item['correct_idx']]}")

    print("\n" + "=" * 50)
    print("DATA PREPARATION COMPLETE")
    print("=" * 50)
    print(f"  ETHICS: data/ethics_sample.csv ({len(ethics)} items)")
    print(f"  MoralChoice: data/moralchoice_sample.csv ({len(mc)} items)")
    print(f"  MORABLES: data/morables/morables_full.csv ({len(morables)} items)")
    print(f"  MORABLES sample: data/morables/morables_sample.csv ({sample_size} items)")


if __name__ == "__main__":
    main()
