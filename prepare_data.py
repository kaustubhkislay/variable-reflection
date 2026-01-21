"""Prepare experimental data samples."""

import pandas as pd
import numpy as np
from pathlib import Path
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

    print("\n" + "=" * 50)
    print("DATA PREPARATION COMPLETE")
    print("=" * 50)
    print(f"  ETHICS: data/ethics_sample.csv ({len(ethics)} items)")
    print(f"  MoralChoice: data/moralchoice_sample.csv ({len(mc)} items)")


if __name__ == "__main__":
    main()
