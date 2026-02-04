import os

# API Configuration (Anthropic)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Anthropic model ID
MODEL = CLAUDE_MODEL  # Default model (backwards compatibility)

# API Configuration (OpenRouter - for Gemini and other models)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_MODEL = "google/gemini-3-flash-preview"  # OpenRouter model ID

# Claude Experiment Parameters
THINKING_BUDGET = 2000
MAX_TOKENS_NO_THINKING = 1000
MAX_TOKENS_WITH_THINKING = 4000
MAX_TOKENS_LEVEL_0 = 30  # Reduced token limit for Level 0 (direct response)
TEMPERATURE = 0  # Deterministic for reproducibility

# Gemini Experiment Parameters
# Gemini experiments use a fixed CoT prompt (level 2) while varying thinking_level
GEMINI_THINKING_LEVELS = ["minimal", "low", "medium", "high"]
GEMINI_PROMPT_LEVEL = 2  # Always use Chain-of-thought prompts for Gemini
GEMINI_MAX_TOKENS = 4000

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
RAW_DIR = "results/raw"
PROCESSED_DIR = "results/processed"

# Experiment Settings
N_RUNS = 3
RANDOM_SEED = 67

# Rate Limiting
CALLS_PER_MINUTE = 50  # Adjust based on your API tier
