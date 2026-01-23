# Variable Reflection

A research framework for studying how different levels of prompted reflection affect Claude's moral reasoning across multiple ethics benchmarks.

## Overview

This project investigates whether prompting large language models to engage in varying degrees of self-reflection improves the quality, consistency, and calibration of their moral judgments. The experiments test Claude across three established moral reasoning benchmarks using six distinct reflection levels.

## Benchmarks

| Benchmark | Task | Categories | Items |
|-----------|------|------------|-------|
| **ETHICS** | Classify scenarios as "wrong" or "not wrong" | commonsense, deontology, virtue | 1500 |
| **MoralChoice** | Choose the more morally acceptable option (A/B) | low ambiguity, high ambiguity | 500 |
| **MORABLES** | Identify the correct moral lesson from Aesop's fables (A-E) | — | 100 |

## Reflection Levels

| Level | Strategy | Description |
|-------|----------|-------------|
| 0 | Direct intuition | Immediate response, no reasoning |
| 1 | Minimal | Simple question reordering |
| 2 | Chain-of-thought | "Think step by step" prompting |
| 3 | Structured analysis | Explicit value weighing for each option |
| 4 | Devil's advocate | Must argue against initial intuition |
| 5 | Two-pass reflection | Answer → then challenge own reasoning |

## Project Structure

```
variable-reflection/
├── config.py              # API and experiment configuration
├── prompts.py             # Reflection-level prompt templates
├── run_experiment.py      # Main experiment runner (sync/async)
├── run_pilot.py           # Quick pilot testing
├── prepare_data.py        # Data sampling and preparation
├── analyze_results.py     # Results analysis
├── src/
│   ├── api.py             # Claude API wrapper with rate limiting
│   ├── extraction.py      # Answer extraction from responses
│   ├── metrics.py         # Evaluation metrics
│   ├── analysis.py        # Statistical analysis utilities
│   └── visualize.py       # Plotting functions
├── data/
│   ├── ethics/            # ETHICS benchmark data
│   ├── morables/          # MORABLES benchmark data
│   ├── moralchoice/       # MoralChoice benchmark data
│   └── *_sample.csv       # Sampled subsets for experiments
├── results/
│   ├── raw/               # Raw API responses
│   ├── processed/         # Cleaned results
│   └── pilot/             # Pilot run outputs
├── outputs/
│   └── tables/            # Generated analysis tables
└── notebooks/
    └── run_experiments.ipynb  # Interactive experiment notebook
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/variable-reflection.git
cd variable-reflection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

## Configuration

Edit `config.py` to adjust experiment parameters:

```python
MODEL = "claude-haiku-4-5-20251001"  # Model to use
THINKING_BUDGET = 2000               # Extended thinking token budget
MAX_TOKENS_NO_THINKING = 1000        # Response limit without thinking
MAX_TOKENS_WITH_THINKING = 4000      # Response limit with thinking
MAX_TOKENS_LEVEL_0 = 30              # Minimal tokens for Level 0
TEMPERATURE = 0                       # Deterministic responses
N_RUNS = 1                            # Repetitions per condition
CALLS_PER_MINUTE = 50                 # Rate limit
```

## Usage

### Prepare Data

```bash
python prepare_data.py --sample-size 100
```

### Run Experiments

**Command Line (Async - Recommended):**
```bash
python run_experiment.py --mode async --sample-size 50
```

**Command Line (Sync):**
```bash
python run_experiment.py --mode sync --sample-size 50
```

**Jupyter Notebook:**
```bash
jupyter notebook notebooks/run_experiments.ipynb
```

### Quick Pilot Test

```bash
python run_pilot.py --items 5
```

## Experimental Conditions

Each item is tested under multiple conditions:

- **Reflection Levels**: 0, 1, 2, 3, 4, 5 (or subset: 0, 2, 4, 5 recommended)
- **Extended Thinking**: Enabled / Disabled
- **Runs**: Configurable repetitions for consistency analysis

## Metrics Tracked

### Per-Response Metrics

| Metric | Description |
|--------|-------------|
| `extracted_answer` | Parsed answer from response |
| `correct` | Whether answer matches ground truth |
| `confidence` | Self-reported confidence (0-100) |
| `confidence_category` | very_low, low, medium, high, very_high |
| `response_length` | Word count |
| `reasoning_markers` | Count of deliberation phrases |
| `uncertainty_markers` | Count of hedging expressions |
| `thinking_content` | Extended thinking block (if enabled) |
| `input_tokens` / `output_tokens` | Token usage |

### Category Tracking

- **ETHICS**: `subscale` (commonsense, deontology, virtue)
- **MoralChoice**: `ambiguity` (low, high)

## Analysis

```bash
python analyze_results.py
```

Generates:
- Accuracy by level and thinking condition
- Confidence calibration curves
- Consistency across runs
- Category-level breakdowns

## Key Research Questions

1. Does increased reflection improve moral judgment accuracy?
2. How does extended thinking interact with prompted reflection?
3. Is there an optimal reflection level, or diminishing returns?
4. How well-calibrated is model confidence across reflection levels?
5. Do different moral frameworks (deontology vs virtue vs commonsense) respond differently to reflection?

## License

MIT

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{variable_reflection,
  title = {Variable Reflection: Studying Prompted Reflection in LLM Moral Reasoning},
  year = {2025},
  url = {https://github.com/yourusername/variable-reflection}
}
```
