# Metrics and Categories Reference

A comprehensive guide to all subscales, categories, and metrics available for analysis across the three benchmarks.

---

## Benchmarks Overview

| Benchmark | Task | Answer Type | Ground Truth |
|-----------|------|-------------|--------------|
| ETHICS | Classify moral scenarios | `wrong` / `not wrong` | Yes |
| MoralChoice | Choose more acceptable option | `A` / `B` | No (preference) |
| MORABLES | Identify correct moral lesson | `A`-`E` | Yes |

---

## Categories by Benchmark

### ETHICS

Based on the [Aligning AI With Shared Human Values](https://arxiv.org/abs/2008.02275) paper (Hendrycks et al., ICLR 2021).

| Subscale | Description | Moral Framework | Currently Included |
|----------|-------------|-----------------|-------------------|
| **Commonsense** | Everyday moral intuitions about right/wrong | Folk morality | Yes |
| **Deontology** | Rule-based ethics; duties and obligations | Kantian ethics | Yes |
| **Virtue** | Character traits; virtuous vs vicious behavior | Aristotelian ethics | Yes |
| **Justice** | Fairness, impartiality, desert-based judgments | Rawlsian ethics | No |
| **Utilitarianism** | Outcome comparison; which scenario is better/worse | Consequentialism | No |

**Data field**: `subscale`

**Current distribution** (in sample):
- Commonsense: 500 items
- Deontology: 500 items
- Virtue: 500 items

---

### MoralChoice

Based on [Evaluating the Moral Beliefs Encoded in LLMs](https://arxiv.org/abs/2307.14324) (Scherrer et al., NeurIPS 2023).

| Category | Description | N Items |
|----------|-------------|---------|
| **Low Ambiguity** | One action is clearly morally preferred | 687 |
| **High Ambiguity** | Neither action is clearly preferred; genuine dilemma | 680 |

**Data field**: `ambiguity`

**Moral Framework**: Based on Gert's Common Morality with 10 rules:
- *Do not cause harm*: Do not kill, cause pain, disable, deprive of freedom, deprive of pleasure
- *Do not violate trust*: Do not deceive, break promises, cheat, break law, neglect duty

**Additional metadata** (in original dataset):
- `scenario_id`: Unique identifier with prefix indicating rule category
- `context`: The situational description
- `option_a`, `option_b`: The two choices

---

### MORABLES

Based on [MORABLES: A Benchmark for Assessing Abstract Moral Reasoning](https://arxiv.org/abs/2509.12371) (Marcuzzo et al., EMNLP 2025).

| Category | Description |
|----------|-------------|
| **No subcategories** | All items are Aesop's fables with moral inference task |

**Task structure**: Multiple-choice (5 options A-E), one correct moral lesson.

**Data fields**:
- `fable`: The story text
- `option_a` through `option_e`: Five candidate morals
- `correct_idx`: Index of correct answer (0-4)
- `correct_moral`: Text of the correct moral

---

## Experimental Conditions

| Condition | Values | Description |
|-----------|--------|-------------|
| `level` | 0, 2, 4, 5 | Reflection level (see below) |
| `thinking` | True, False | Extended thinking enabled |
| `run` | 0, 1, 2 | Repetition number |

### Reflection Levels

| Level | Strategy | Prompt Style |
|-------|----------|--------------|
| 0 | Direct intuition | No reasoning, immediate answer |
| 2 | Chain-of-thought | "Think step by step" |
| 4 | Devil's advocate | Argue against initial intuition |
| 5 | Two-pass | Answer, then challenge own reasoning |

---

## Metrics Tracked

### Core Metrics (All Benchmarks)

| Metric | Type | Description |
|--------|------|-------------|
| `item_id` | string | Unique item identifier |
| `level` | int | Reflection level (0, 2, 4, 5) |
| `thinking` | bool | Extended thinking enabled |
| `run` | int | Run number (0, 1, 2) |
| `response` | string | Full model response text |
| `thinking_content` | string | Extended thinking block (if enabled) |
| `extracted_answer` | string | Parsed answer from response |
| `timestamp` | datetime | When API call was made |

### Accuracy Metrics (ETHICS, MORABLES)

| Metric | Type | Description |
|--------|------|-------------|
| `correct` | bool | Whether extracted answer matches ground truth |
| `label` | string | Ground truth label (ETHICS only) |
| `correct_idx` | int | Correct answer index (MORABLES only) |
| `correct_answer` | string | Correct answer letter (MORABLES only) |

### Confidence Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `confidence` | int | Self-reported confidence (0-100) |
| `confidence_category` | string | Binned confidence level |

**Confidence categories**:
- `very_low`: 0-20
- `low`: 21-40
- `medium`: 41-60
- `high`: 61-80
- `very_high`: 81-100

### Linguistic Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `response_length` | int | Word count of response |
| `reasoning_markers` | int | Count of deliberation phrases |
| `uncertainty_markers` | int | Count of hedging expressions |

**Reasoning markers** (counted):
> however, on the other hand, alternatively, but, although, that said, nonetheless, one could argue, some might say, it's possible, i'm not certain, arguably, perhaps, let me reconsider, wait, actually, from another perspective, considering, weighing, on reflection, thinking about, upon further, that being said, having said that, nevertheless

**Uncertainty markers** (counted):
> might, could, perhaps, possibly, uncertain, not sure, debatable, arguably, it depends, hard to say, difficult to judge, unclear, may be, potentially, not certain, unsure, i think, i believe, in my opinion, seems, appears to, likely, unlikely, probably

### Token Usage

| Metric | Type | Description |
|--------|------|-------------|
| `input_tokens` | int | Tokens in prompt |
| `output_tokens` | int | Tokens in response |

---

## Analysis Dimensions

### Primary Analyses

| Analysis | Applicable Benchmarks | Key Metrics |
|----------|----------------------|-------------|
| Accuracy by level | ETHICS, MORABLES | `correct`, `level` |
| Accuracy by thinking | ETHICS, MORABLES | `correct`, `thinking` |
| Accuracy by category | ETHICS (subscale), MORABLES | `correct`, `subscale` |
| Choice distribution | MoralChoice | `extracted_answer`, `ambiguity` |
| Consistency across runs | All | `extracted_answer`, `run` |

### Secondary Analyses

| Analysis | Description | Key Metrics |
|----------|-------------|-------------|
| Confidence calibration | Correlation between confidence and accuracy | `confidence`, `correct` |
| Response characteristics | How reflection affects verbosity | `response_length`, `level` |
| Reasoning depth | Deliberation markers by condition | `reasoning_markers`, `level` |
| Uncertainty expression | Hedging by condition | `uncertainty_markers`, `level` |
| Token efficiency | Cost analysis | `input_tokens`, `output_tokens` |

### Cross-Benchmark Comparisons

| Comparison | Description |
|------------|-------------|
| Thinking effect | Does extended thinking help/hurt across benchmarks? |
| Optimal level | Is there a consistent best reflection level? |
| Calibration | Which benchmark shows best confidence calibration? |
| Consistency | Which benchmark has most stable answers across runs? |

---

## Data Access

### Loading Data

```python
import pandas as pd

# Load checkpoints
ethics = pd.read_csv("results/raw/ethics_checkpoint.csv")
mc = pd.read_csv("results/raw/moralchoice_checkpoint.csv")
morables = pd.read_csv("results/raw/morables_checkpoint.csv")

# Filter by condition
level_0 = ethics[ethics['level'] == 0]
with_thinking = ethics[ethics['thinking'] == True]
commonsense = ethics[ethics['subscale'] == 'commonsense']
```

### Common Queries

```python
# Accuracy by level and thinking
ethics.groupby(['level', 'thinking'])['correct'].mean()

# Confidence by subscale
ethics.groupby('subscale')['confidence'].mean()

# Choice distribution by ambiguity
mc.groupby(['ambiguity', 'extracted_answer']).size()

# Consistency: items with same answer across all runs
consistency = ethics.groupby(['item_id', 'level', 'thinking'])['extracted_answer'].nunique()
consistent_items = (consistency == 1).mean()
```

---

## References

1. Hendrycks, D., et al. (2021). *Aligning AI With Shared Human Values*. ICLR 2021. [arXiv:2008.02275](https://arxiv.org/abs/2008.02275)

2. Scherrer, N., et al. (2023). *Evaluating the Moral Beliefs Encoded in LLMs*. NeurIPS 2023. [arXiv:2307.14324](https://arxiv.org/abs/2307.14324)

3. Marcuzzo, M., et al. (2025). *MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables*. EMNLP 2025. [arXiv:2509.12371](https://arxiv.org/abs/2509.12371)
