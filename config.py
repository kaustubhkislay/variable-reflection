import os

# API Configuration (Anthropic)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Anthropic model ID
CLAUDE_OPUS_MODEL = "claude-opus-4-5-20250929"  # Anthropic model ID
MODEL = CLAUDE_MODEL  # Default model (backwards compatibility)

# API Configuration (OpenRouter - for Gemini and other models)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_MODEL = "google/gemini-3-flash-preview"  # OpenRouter model ID
GEMINI_PRO_MODEL = "google/gemini-3-pro-preview"  # OpenRouter model ID
GPT_MODEL = "openai/gpt-5.2"  # OpenRouter model ID

# Claude Experiment Parameters
THINKING_BUDGET = 2000
MAX_TOKENS_NO_THINKING = 1000
MAX_TOKENS_WITH_THINKING = 4000
MAX_TOKENS_LEVEL_0 = 30  # Reduced token limit for Level 0 (direct response)
TEMPERATURE = 0  # Deterministic for reproducibility

# Gemini Experiment Parameters
# Gemini experiments vary both prompt level and thinking_level
GEMINI_THINKING_LEVELS = ["minimal", "low", "medium", "high"]
GEMINI_PROMPT_LEVELS = [0, 2, 4, 5]  # All prompting styles
GEMINI_MAX_TOKENS = 4000
GEMINI_N_RUNS = 1  # Number of runs for Gemini experiments

# GPT Experiment Parameters
GPT_THINKING_LEVELS = ["none", "low", "medium", "high", "xhigh"]
GPT_PROMPT_LEVELS = [0, 2, 4, 5]  # All prompting styles
GPT_MAX_TOKENS = 4000
GPT_N_RUNS = 1  # Number of runs for GPT experiments

# Qwen Experiment Parameters
# Qwen experiments vary both prompt level and reasoning token budget
QWEN_4B_MODEL = "qwen/qwen3-4b"      # OpenRouter model ID
QWEN_14B_MODEL = "qwen/qwen3-14b"    # OpenRouter model ID
QWEN_32B_MODEL = "qwen/qwen3-32b"    # OpenRouter model ID
QWEN_235B_MODEL = "qwen/qwen3-235b"  # OpenRouter model ID
QWEN_THINKING_LEVELS = [1024, 2048, 4096, 8192, 16384, 32768]  # Reasoning token budgets
QWEN_PROMPT_LEVELS = [0, 2, 4, 5]  # All prompting styles
QWEN_MAX_TOKENS = 4000
QWEN_N_RUNS = 1  # Number of runs for Qwen experiments

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
RAW_DIR = "results/raw"
PROCESSED_DIR = "results/processed"

# Experiment Settings
N_RUNS = 3
RANDOM_SEED = 67

# Judge Configuration (LLM-as-Judge for flip-flop detection)
JUDGE_THINKING_LEVEL = "minimal"  # Fast structured output
JUDGE_MAX_TOKENS = 500  # Structured response is compact
JUDGE_MIN_TRACE_WORDS = 20  # Minimum words to consider judgeable

# But-Wait Continuation Technique
BUT_WAIT_SUFFIX = "\n\nbut wait: "
BUT_WAIT_MIN_BUDGET = 100  # Stop when remaining tokens < this

# Rate Limiting
CALLS_PER_MINUTE = 50  # Adjust based on your API tier
