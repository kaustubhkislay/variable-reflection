# Thinking Budget LTS

Experiments measuring LLM inference behavior (latency, throughput, token usage) across RunPod vLLM serverless endpoints with varying parameters.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your RUNPOD_API_KEY
python run_experiment.py --model qwen3-32b "What is the capital of France?"
```

## CLI Usage

```bash
# Basic non-streaming request
python run_experiment.py -m qwen3-32b "Your prompt here"

# Disable thinking
python run_experiment.py -m qwen3-32b --no-think "Your prompt here"

# Streaming with custom temperature and token limit
python run_experiment.py -m qwen3-32b --stream --max-tokens 512 --t 0.3 "Explain quantum computing"

# Save result as JSON with streaming
python run_experiment.py -m qwen3-32b --stream --max-tokens 1000 --json-out results/test1.json "How many r's are there in strawberry?"

# Check if an endpoint is alive and model name is correct
python run_experiment.py -m qwen3-32b --ping
```


## Models available

Found in `endpoints.json`:
- `qwen/qwen3-32b`
- `qwen/qwen3-14b`
- `qwen/qwen3-8b`
- `qwen/qwen3-4b`



## Endpoint Configuration

RunPod vLLM environment variable overrides applied to our endpoints:

| Variable | Default | Our Max Setting | Reason |
|---|---|---|---|
| `MAX_MODEL_LEN` | `0` (auto → 40960) | `32768` | Reduce VRAM usage to fit within worker GPU memory |
| `ENFORCE_EAGER` | `false` | `true` | Disable CUDA graph capture to avoid OOM during init |