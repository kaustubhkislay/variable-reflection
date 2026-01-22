# Integrating MORABLES into Variable-Reflection Experiment

This document outlines how to integrate the [MORABLES benchmark](https://huggingface.co/datasets/cardiffnlp/Morables) into the existing variable-reflection experiment framework.

## Overview

### What is MORABLES?

MORABLES is a benchmark for assessing **abstract moral reasoning** in LLMs using fables and short stories from historical literature. Unlike ETHICS (judging explicit scenarios) or MoralChoice (choosing between options), MORABLES tests the ability to **infer implicit moral lessons** from narratives.

| Aspect | ETHICS | MoralChoice | MORABLES |
|--------|--------|-------------|----------|
| Task | Judge scenario | Choose A or B | Identify moral |
| Reasoning | Explicit judgment | Comparative ethics | Abstract inference |
| Format | Binary (wrong/not wrong) | Binary (A/B) | 5-way multiple choice |
| Ground truth | Yes | No (preference) | Yes |
| Size | 1500 items | 500 items | 709 fables |

### Why Add MORABLES?

1. **Tests different cognitive skill**: Extracting implicit morals vs. judging explicit scenarios
2. **Relevant prior finding**: MORABLES paper found reasoning-enhanced models don't help—directly tests our hypothesis
3. **Built-in adversarial variants**: Can compare to our Level 4 adversarial prompts
4. **Self-contradiction metric**: ~20% self-contradiction rate aligns with our consistency measurements
5. **Ground truth available**: Enables accuracy measurement like ETHICS

---

## Dataset Details

### Access

```python
# Via Hugging Face
from datasets import load_dataset
morables = load_dataset("cardiffnlp/Morables")
```

**Direct URL**: https://huggingface.co/datasets/cardiffnlp/Morables

### Structure

Each entry contains:

| Field | Description |
|-------|-------------|
| `fable` | The narrative text (avg 133 words, 5.6 sentences) |
| `moral` | Ground truth moral statement (avg 11.6 words) |
| `options` | 5 multiple-choice options (1 correct + 4 distractors) |

### Distractor Types

The benchmark uses systematically constructed distractors:

1. **Similar-character**: Morals from different fables with similar characters
2. **Trait-injected**: Original moral modified with character features
3. **Feature-based**: LLM-generated from character traits only
4. **Partial-story**: LLM-generated from initial narrative excerpt (most common error mode)

### Task Variants

| Variant | Description | Items | Use Case |
|---------|-------------|-------|----------|
| **Core** | Standard 5-way MCQA | 709 | Primary evaluation |
| **TF** | True/False per option | 3,545 | Binary classification |
| **NOTO** | "None of the Other" option added | 709 | Tests overconfidence |
| **ADV** | Adversarial modifications | 709 | Robustness testing |

**Recommendation**: Use **Core** for primary experiment, **ADV** for robustness analysis.

---

## Integration Plan

### 1. Data Preparation

Create `prepare_morables_data.py`:

```python
"""Prepare MORABLES data for experiment."""

from datasets import load_dataset
import pandas as pd
from pathlib import Path

def prepare_morables():
    """Load and prepare MORABLES dataset."""

    # Load from Hugging Face
    dataset = load_dataset("cardiffnlp/Morables")

    # Convert to DataFrame
    df = pd.DataFrame(dataset['test'])  # or appropriate split

    # Add item IDs
    df['item_id'] = [f"morables_{i}" for i in range(len(df))]

    # Ensure consistent column naming
    # Adjust based on actual HF column names
    df = df.rename(columns={
        'story': 'fable',        # if needed
        'label': 'correct_idx',  # index of correct answer
    })

    # Save
    Path("data/morables").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/morables/morables_sample.csv", index=False)

    print(f"Prepared {len(df)} MORABLES items")
    return df

if __name__ == "__main__":
    prepare_morables()
```

### 2. Prompt Templates

Add to `prompts.py`:

```python
# =============================================================================
# MORABLES PROMPTS
# =============================================================================

MORABLES_PROMPTS = {
    0: """Read this fable:

{fable}

What is the moral of this story?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

Respond with exactly one letter: A, B, C, D, or E.""",

    1: """Read this fable:

{fable}

What is the moral of this story?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

State your answer clearly.""",

    2: """Read this fable:

{fable}

What is the moral of this story?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

Think step by step about what lesson the story teaches, then state your final answer on a new line as: "Final answer: X" where X is A, B, C, D, or E.""",

    3: """Read this fable:

{fable}

What is the moral of this story?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

Before answering:
1. Identify the key characters and their actions
2. Consider what consequence or lesson emerges from the narrative
3. Evaluate which moral best captures the story's message

Then state your final answer: A, B, C, D, or E.""",

    4: """Read this fable:

{fable}

What is the moral of this story?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

Before deciding:
1. What is your initial intuition about the moral?
2. Which other options might also seem plausible? Why?
3. What distinguishes the true moral from surface-level interpretations?
4. Does reconsidering change your answer?

Provide your final answer: A, B, C, D, or E.""",

    "5_pass1": """Read this fable:

{fable}

What is the moral of this story?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

Explain your reasoning, then clearly state your final answer on a new line as: "Final answer: X" where X is A, B, C, D, or E.""",

    "5_pass2": """You previously analyzed the fable and answered:

{previous_response}

Now reflect on your reasoning:
1. Did you consider the FULL narrative arc, not just the beginning?
2. Could any distractor be a surface-level interpretation?
3. Does the moral truly capture what the story teaches?

State your final answer: A, B, C, D, or E."""
}


def get_morables_prompt(level: int, fable: str, options: list,
                        previous_response: str = None) -> str:
    """
    Generate MORABLES prompt for given level.

    Args:
        level: Reflection level (0-5)
        fable: The fable/story text
        options: List of 5 moral options [A, B, C, D, E]
        previous_response: Response from pass 1 (for level 5 pass 2)

    Returns:
        Formatted prompt string
    """
    option_a, option_b, option_c, option_d, option_e = options

    if level == 5 and previous_response is None:
        return MORABLES_PROMPTS["5_pass1"].format(
            fable=fable,
            option_a=option_a, option_b=option_b, option_c=option_c,
            option_d=option_d, option_e=option_e
        )
    elif level == 5 and previous_response is not None:
        return MORABLES_PROMPTS["5_pass2"].format(previous_response=previous_response)
    else:
        return MORABLES_PROMPTS[level].format(
            fable=fable,
            option_a=option_a, option_b=option_b, option_c=option_c,
            option_d=option_d, option_e=option_e
        )
```

### 3. Answer Extraction

Add to `src/extraction.py`:

```python
def extract_morables_answer(response: str) -> Optional[str]:
    """
    Extract A/B/C/D/E from MORABLES response.

    Returns:
        "A", "B", "C", "D", "E", or None if extraction failed
    """
    if response is None:
        return None

    text = response.strip()
    valid_answers = {"A", "B", "C", "D", "E"}

    # Direct match (Level 0)
    if text.upper() in valid_answers:
        return text.upper()

    # High priority: "Final answer:" patterns
    final_patterns = [
        r'final answer[:\s]*["\']?\**([A-E])\**["\']?',
        r'\*\*final answer[:\s]*([A-E])\*\*',
    ]
    for pattern in final_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Medium priority: explicit choice patterns
    choice_patterns = [
        r'(?:I )?(?:choose|select|pick|go with)[:\s]*(?:option\s*)?["\']?([A-E])["\']?',
        r'(?:my )?(?:answer|choice)[:\s]*["\']?([A-E])["\']?',
        r'(?:the )?(?:moral|answer|correct option)\s*(?:is|seems to be)[:\s]*(?:option\s*)?([A-E])',
        r'(?:therefore|thus|hence)[,\s]*(?:option\s*)?([A-E])',
        r'(?:option\s+)?([A-E])\s*(?:is|captures|represents)\s*(?:the)?\s*(?:correct|best|true)',
    ]

    for pattern in choice_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).upper() in valid_answers:
            return match.group(1).upper()

    # Check last 3 lines for answer
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in reversed(lines[-3:] if len(lines) >= 3 else lines):
        if line.startswith('#'):
            continue
        line_match = re.search(r'(?:option\s*)?([A-E])\s*[.!]?\s*$', line, re.IGNORECASE)
        if line_match:
            return line_match.group(1).upper()

    # Last resort: find last standalone A-E
    choice_indicators = list(re.finditer(r'(?:^|\s|:)([A-E])(?:\s*[.!,)]|\s*$)', text, re.IGNORECASE))
    if choice_indicators:
        return choice_indicators[-1].group(1).upper()

    return None
```

### 4. Experiment Runner

Create `run_morables_experiment.py` or add to `run_experiment.py`:

```python
def run_single_item_morables(row, level, thinking):
    """Run single MORABLES item at given condition."""

    # Parse options (adjust based on actual data format)
    options = [row['option_a'], row['option_b'], row['option_c'],
               row['option_d'], row['option_e']]

    if level == 5:
        # Two-pass
        prompt1 = get_morables_prompt(5, row['fable'], options)
        response1 = call_with_rate_limit(prompt1, thinking)

        prompt2 = get_morables_prompt(5, row['fable'], options, response1.content)
        response2 = call_with_rate_limit(prompt2, thinking)

        return {
            'content': response2.content,
            'thinking': response2.thinking,
            'full_response': f"[PASS1]\n{response1.content}\n\n[PASS2]\n{response2.content}",
            'input_tokens': response1.input_tokens + response2.input_tokens,
            'output_tokens': response1.output_tokens + response2.output_tokens,
        }
    else:
        prompt = get_morables_prompt(level, row['fable'], options)
        response = call_with_rate_limit(prompt, thinking)

        return {
            'content': response.content,
            'thinking': response.thinking,
            'full_response': response.content,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }


def run_morables_experiment():
    """Run full MORABLES experiment."""

    morables = pd.read_csv("data/morables/morables_sample.csv")

    if SAMPLE_SIZE:
        morables = morables.sample(n=min(SAMPLE_SIZE, len(morables)),
                                    random_state=config.RANDOM_SEED)
        print(f"  Using {len(morables)} items")

    results = []

    for run in range(N_RUNS):
        for thinking in THINKING_CONDITIONS:
            for level in LEVELS:
                thinking_label = "ON" if thinking else "OFF"
                print(f"MORABLES Run {run+1}, Level {level}, Thinking {thinking_label}")

                for _, row in tqdm(morables.iterrows(), total=len(morables)):
                    try:
                        response_data = run_single_item_morables(row, level, thinking)
                        extracted = extract_morables_answer(response_data['content'])

                        # Map extracted letter to index for correctness check
                        letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                        extracted_idx = letter_to_idx.get(extracted)
                        correct = extracted_idx == row['correct_idx'] if extracted_idx is not None else None

                        results.append({
                            'item_id': row['item_id'],
                            'fable': row['fable'][:200],  # Truncate
                            'correct_answer': row['correct_idx'],
                            'level': level,
                            'thinking': thinking,
                            'run': run,
                            'response': response_data['full_response'],
                            'thinking_content': response_data['thinking'],
                            'extracted_answer': extracted,
                            'correct': correct,
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
                        })

                # Checkpoint
                pd.DataFrame(results).to_csv("results/raw/morables_checkpoint.csv", index=False)

    return pd.DataFrame(results)
```

### 5. Analysis Extensions

Add to `src/analysis.py`:

```python
def compute_morables_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy for MORABLES by level and thinking condition."""

    # Filter to valid extractions
    valid = df[df['extracted_answer'].notna()].copy()

    accuracy = valid.groupby(['level', 'thinking']).agg(
        accuracy=('correct', 'mean'),
        n=('correct', 'count'),
        se=('correct', lambda x: x.std() / (len(x) ** 0.5))
    ).reset_index()

    return accuracy


def compute_morables_distractor_analysis(df: pd.DataFrame,
                                          distractor_types: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which distractor types models fall for most often.

    This is unique to MORABLES and tests if reflection helps avoid
    surface-level interpretations (partial-story distractors).
    """
    # Join with distractor type information
    merged = df.merge(distractor_types, on='item_id')

    # For incorrect answers, identify which distractor type was chosen
    incorrect = merged[merged['correct'] == False]

    distractor_errors = incorrect.groupby(['level', 'thinking', 'distractor_type']).size()

    return distractor_errors.reset_index(name='count')
```

---

## Expected Outcomes

### Hypotheses to Test

1. **Does reflection reduce partial-story errors?**
   - MORABLES paper found models over-rely on initial narrative cues
   - Levels 2-5 should encourage full-story consideration

2. **Does adversarial reflection (Level 4) help with adversarial variants?**
   - Compare Level 4 performance on Core vs ADV variants

3. **Does extended thinking help abstract inference differently than judgment?**
   - May see different thinking × level interaction than ETHICS/MoralChoice

4. **Is self-contradiction rate affected by reflection level?**
   - MORABLES reports ~20% self-contradiction
   - Multi-run design can measure this

### New Metrics

| Metric | Description |
|--------|-------------|
| Accuracy by distractor type | Which distractors fool the model? |
| Partial-story error rate | How often does model use only beginning of story? |
| Self-contradiction rate | Answer changes across identical prompts |
| Thinking utilization | Does model use full thinking budget? |

---

## File Structure After Integration

```
variable-reflection/
├── data/
│   ├── ethics_sample.csv
│   ├── moralchoice_sample.csv
│   └── morables/
│       └── morables_sample.csv      # NEW
├── prompts.py                        # Add MORABLES_PROMPTS
├── src/
│   ├── extraction.py                 # Add extract_morables_answer()
│   └── analysis.py                   # Add morables-specific analysis
├── prepare_morables_data.py          # NEW
├── run_experiment.py                 # Add run_morables_experiment()
└── results/
    ├── raw/
    │   └── morables_checkpoint.csv   # NEW
    └── processed/
        └── morables_results.csv      # NEW
```

---

## Implementation Checklist

- [ ] Download dataset from HuggingFace: `cardiffnlp/Morables`
- [ ] Create `prepare_morables_data.py`
- [ ] Add prompt templates to `prompts.py`
- [ ] Add `extract_morables_answer()` to `src/extraction.py`
- [ ] Add experiment runner function
- [ ] Add analysis functions for 5-way MCQA
- [ ] Run pilot test (~10 items)
- [ ] Validate extraction accuracy
- [ ] Run full experiment
- [ ] Analyze distractor-type error patterns

---

## Estimated Costs

| Configuration | Items | API Calls | Est. Cost (Haiku) |
|--------------|-------|-----------|-------------------|
| Pilot | 10 | ~840 | ~$0.50 |
| Full (100 items) | 100 | ~8,400 | ~$5 |
| Full (709 items) | 709 | ~59,556 | ~$35 |

*Assumes 6 levels × 2 thinking × 3 runs, Level 5 = 2 calls*

---

## References

- [MORABLES Paper (arXiv)](https://arxiv.org/abs/2509.12371)
- [MORABLES Dataset (HuggingFace)](https://huggingface.co/datasets/cardiffnlp/Morables)
- [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1411/)
