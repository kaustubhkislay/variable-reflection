import os

# API Configuration (OpenRouter)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-4.5-haiku"  # OpenRouter model format

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
N_RUNS = 3
RANDOM_SEED = 42

# Rate Limiting
CALLS_PER_MINUTE = 50  # Adjust based on your API tier
