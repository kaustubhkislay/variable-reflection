import os

# API Configuration (Anthropic)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-haiku-4-5-20251001"  # Anthropic model ID

# Experiment Parameters
THINKING_BUDGET = 2000
MAX_TOKENS_NO_THINKING = 1000
MAX_TOKENS_WITH_THINKING = 4000
TEMPERATURE = 0  # Deterministic for reproducibility

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
RAW_DIR = "results/raw"
PROCESSED_DIR = "results/processed"

# Experiment Settings
N_RUNS = 1
RANDOM_SEED = 67

# Rate Limiting
CALLS_PER_MINUTE = 50  # Adjust based on your API tier
