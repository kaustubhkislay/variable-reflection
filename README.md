# Thinking Budget LTS

Experiments measuring LLM inference behavior (latency, throughput, token usage) across RunPod vLLM serverless endpoints with varying parameters.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your RUNPOD_API_KEY
python run_experiment.py --model qwen3-8b "What is the capital of France?"
```

## Models

| Alias | Model |
|---|---|
| `qwen3-4b` | `qwen/qwen3-4b` |
| `qwen3-8b` | `qwen/qwen3-8b` |
| `qwen3-14b` | `qwen/qwen3-14b` |
| `qwen3-32b` | `qwen/qwen3-32b` |

## run_experiment.py — smoke tests

```bash
# Streaming + JSON output for each model
python run_experiment.py -m qwen3-4b  --stream --max-tokens 200 --json-out results/smoke_4b.json  "Say hello"
python run_experiment.py -m qwen3-8b  --stream --max-tokens 200 --json-out results/smoke_8b.json  "Say hello"
python run_experiment.py -m qwen3-14b --stream --max-tokens 200 --json-out results/smoke_14b.json "Say hello"
python run_experiment.py -m qwen3-32b --stream --max-tokens 200 --json-out results/smoke_32b.json "Say hello"

# Endpoint health check (validates model name + liveness)
python run_experiment.py -m qwen3-4b  --ping
python run_experiment.py -m qwen3-8b  --ping
python run_experiment.py -m qwen3-14b --ping
python run_experiment.py -m qwen3-32b --ping
```

## run_experiments_prefills.py — prefill demo

Runs the default 5-prefill battery (`no_prefill`, `cheeky_think`, `think_close`, `think_seeded`, `answer_steer`) and saves results to JSON.

```bash
# Demo run — outputs results/prefill_demo.json
python run_experiments_prefills.py -m qwen3-8b --json-out results/prefill_demo.json

# Single custom prefill
python run_experiments_prefills.py -m qwen3-8b --prefill "</think>" "Explain gravity"
```

## Endpoint Configuration

RunPod vLLM environment variable overrides applied to our endpoints:

| Variable | Default | Our Setting | Reason |
|---|---|---|---|
| `MAX_MODEL_LEN` | `0` (auto → 40960) | `32768` | Reduce VRAM usage |
| `ENFORCE_EAGER` | `false` | `true` | Disable CUDA graph capture to avoid OOM during init |
