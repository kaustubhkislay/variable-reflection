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
├── run_experiment.py      # Main experiment runner (sync/async with resume)
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
│   ├── raw/               # Checkpoints during experiment
│   ├── processed/         # Final cleaned results
│   └── pilot/             # Pilot run outputs
├── outputs/               # Analysis outputs
│   ├── *.png              # Generated figures
│   └── *.csv              # Summary tables
└── notebooks/
    └── analysis.ipynb     # Results analysis notebook
```

## Installation

```bash
# Clone the repository
git clone https://github.com/kaustubhkislay/variable-reflection.git
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
N_RUNS = 3                            # Repetitions per condition
CALLS_PER_MINUTE = 50                 # Rate limit
RANDOM_SEED = 67                      # For reproducible sampling
```

## Usage

### Prepare Data

```bash
python prepare_data.py --sample-size 100
```

### Run Experiments

**Async Mode (Recommended):**
```bash
python run_experiment.py --async --sample 100
```

**Resume After Interruption:**
```bash
python run_experiment.py --async --resume --sample 100
```

**Sync Mode:**
```bash
python run_experiment.py --sample 100
```

**Run Specific Benchmarks:**
```bash
python run_experiment.py --async --ethics --moralchoice --sample 100
```

### Long-Running Experiments

For experiments lasting several hours, use `tmux` to keep the process running:

```bash
tmux new -s experiment
python run_experiment.py --async --sample 100
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t experiment
```

### Quick Pilot Test

```bash
python run_experiment.py --async --sample 6
```

### Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Experimental Conditions

Each item is tested under multiple conditions:

- **Reflection Levels**: 0, 2, 4, 5 (default; captures distinct cognitive strategies)
- **Extended Thinking**: Enabled / Disabled
- **Runs**: 3 repetitions for consistency analysis

### Stratified Sampling

Samples are stratified to maintain category balance:
- **ETHICS**: Equal items from commonsense, deontology, virtue subscales
- **MoralChoice**: Equal items from low and high ambiguity conditions
- **MORABLES**: Random sample (no categories)

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

Open the analysis notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook generates:
- Accuracy by level and thinking condition (with visualizations)
- Confidence calibration curves
- Consistency across runs
- Category-level breakdowns (subscale, ambiguity)
- Response characteristics (length, reasoning markers)
- Summary statistics exported to CSV

Outputs are saved to the `outputs/` directory.


## License

MIT

```
