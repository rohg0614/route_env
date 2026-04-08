---
title: Route Env Environment Server
emoji: 📸
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Route Env Environment

A real-world RL benchmark for **single-driver ride dispatch and repositioning** on a stochastic city graph.

The driver must trade off:
- immediate fare collection,
- empty-mile burn,
- proactive repositioning,
- SLA lateness risk under changing demand.

## Why this environment is useful

This models a practical operations task: where should an idle driver go now to maximize long-term utility under uncertainty?

It is designed for evaluating policy reasoning under:
- non-stationary demand (time-varying Poisson arrivals),
- constrained actions (adjacent-node reposition only),
- dense but multi-objective reward shaping,
- realistic tradeoffs (profit vs. efficiency vs. punctuality).

## Action and Observation Spaces

### Action (`RouteAction`)
- `action_type`: `wait | accept_ride | reposition`
- `ride_id`: required when accepting a ride
- `target_node`: required when repositioning (must be adjacent)

### Observation (`RouteObservation`)
- `task_name`: `easy | medium | hard`
- `current_node`
- `time_of_day_sin`, `time_of_day_cos`
- `driver_status`: `idle | busy | en_route`
- `shift_hours_remaining`
- `live_demand_matrix`
- `available_rides`
- `last_action_error`
- `normalized_progress_score` (grader output in `[0.0, 1.0]`)

## Reward Function

Per-step reward:

`R_t = 1.0*(fare/max_fare) - 0.9*(empty_distance/max_distance) - 0.15*did_reposition + 1_ride*(2.0*exp(-0.1*waiting_time)) - 1.5*late_penalty`

This enforces:
- profit signal,
- operational-cost penalty,
- anti-jitter reposition friction,
- urgency bonus for timely pickups,
- SLA lateness punishment.

## Tasks and Difficulty Range

Task presets are explicit in `tasks.py`:
- `easy`
- `medium`
- `hard`

Each task changes graph size, horizon, demand intensity, and constraints.

The environment supports:
- default cyclic task selection on reset, and
- explicit task selection: `reset(task_name="hard", seed=123)`.

## Programmatic Grader (`[0.0, 1.0]`)

The grader is explicit and deterministic in `grader.py`:
- input: `step_count`, `completed_rides`, `late_rides`, `total_reward`
- output: bounded scalar score in `[0.0, 1.0]`

The score is exposed in every observation as `normalized_progress_score`.

## Determinism and Reproducibility

- Global deterministic seed via env var: `SEED` (default `42`)
- Optional per-episode seed override: `reset(seed=...)`
- Baselines can produce reproducible scores by fixing seeds.

## Baseline Scripts

### 1) Hackathon inference script (OpenAI client + required output format)
`inference.py`

Supports:
- `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- optional OpenLLM policy mode via `USE_OPENLLM_AGENT=true`
- strict judge-safe mode by default: `STRICT_SINGLE_EPISODE=true`
- optional long-run trajectories (only when strict mode is disabled)

### 2) Deterministic benchmark script
`baseline_benchmark.py`

Runs heuristic policy across easy/medium/hard and prints mean score per task.

```bash
python baseline_benchmark.py
```
This also writes `baseline_scores.json` for reproducibility evidence.

## Judge Runbook (Strict)

Use these exact commands for strict validation:

```bash
# 1) Validate spec
openenv validate --verbose

# 2) Start docker env
docker build -t route_env-env:latest -f server/Dockerfile .
docker rm -f route_env_local || true
docker run -d --name route_env_local -p 7860:7860 route_env-env:latest

# 3) Reproducible benchmark artifact
SEED=42 ENV_BASE_URL=http://localhost:7860 python baseline_benchmark.py

# 4) Strict baseline inference (single episode output contract)
STRICT_SINGLE_EPISODE=true ENV_BASE_URL=http://localhost:7860 python inference.py
```

## Setup

```bash
uv sync
openenv validate --verbose
```

## Local run (Docker)

```bash
docker build -t route_env-env:latest -f server/Dockerfile .
docker run -d --name route_env_local -p 7860:7860 route_env-env:latest
```

Endpoints:
- Web UI: `http://localhost:7860/web/`
- Docs: `http://localhost:7860/docs`
- Health: `http://localhost:7860/health`

## Hugging Face deployment

```bash
openenv push --repo-id <username/space-name>
```

Space metadata:
- docker sdk
- app port `7860`
- web base path `/web`

## Project Structure

```
route_env/
├── __init__.py
├── client.py
├── models.py
├── tasks.py
├── grader.py
├── inference.py
├── baseline_benchmark.py
├── openenv.yaml
├── pyproject.toml
├── uv.lock
└── server/
    ├── app.py
    ├── route_env_environment.py
    └── Dockerfile
```
