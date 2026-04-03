---
title: Route Env Environment Server
emoji: 📸
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Route Env Environment

A graph-based ride dispatch optimization environment for OpenEnv. The agent controls one driver that can wait, accept rides, or reposition to adjacent zones while demand changes stochastically over time.

## Quick Start

The simplest way to use the Route Env environment is through the `RouteEnv` class:

```python
from route_env import RouteAction, RouteEnv

try:
    # Create environment from Docker image
    route_envenv = RouteEnv.from_docker_image("route_env-env:latest")

    # Reset
    result = route_envenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = route_envenv.step(RouteAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    route_envenv.close()
```

That's it! The `RouteEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t route_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action Space
`RouteAction` supports:
- `wait`
- `accept_ride` with `ride_id`
- `reposition` with `target_node` (adjacent node only)

### Observation Space
`RouteObservation` provides:
- `current_node`
- `time_of_day_sin`, `time_of_day_cos`
- `driver_status`
- `shift_hours_remaining`
- `live_demand_matrix`
- `available_rides`
- `last_action_error`
- `normalized_progress_score`

### Reward Function
Dense reward per step:

`R_t = 1.0*(fare/max_fare) - 0.9*(empty_distance/max_distance) - 0.15*did_reposition + 1_ride*(2.0*exp(-0.1*waiting_time)) - 1.5*late_penalty`

This balances profit, empty-mile burn, anti-jitter friction, urgency-aware pickup bonus, and SLA lateness penalties.

## Tasks and Graders

The environment cycles through three deterministic task presets on each reset:
- `easy` (small graph, shorter horizon)
- `medium`
- `hard` (larger graph, longer horizon)

A programmatic grader computes `normalized_progress_score` in `[0,1]` using ride efficiency, punctuality, and reward quality.

## Long-Running Multi-Trajectory Evaluation

The baseline runner in `inference.py` natively supports long-running evaluation over multiple trajectories.

```bash
# Runs 25 trajectories; each can run up to 300 steps
NUM_TRAJECTORIES=25 MAX_STEPS_PER_TRAJECTORY=300 python inference.py
```

Notes:
- Output format remains compliant: each trajectory emits only `[START]`, `[STEP]`, `[END]`.
- `NUM_TRAJECTORIES=1` is the default for strict single-episode baseline checks.
- This is intended for judge-visible robustness and variance testing across stochastic trajectories.

## Advanced Usage

### Connecting to an Existing Server

If you already have a Route Env environment server running, you can connect directly:

```python
from route_env import RouteEnv

# Connect to existing server
route_envenv = RouteEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = route_envenv.reset()
result = route_envenv.step(RouteAction(message="Hello!"))
```

Note: When connecting to an existing server, `route_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from route_env import RouteAction, RouteEnv

# Connect with context manager (auto-connects and closes)
with RouteEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(RouteAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    RouteEnvironment,  # Pass class, not instance
    RouteAction,
    RouteObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from route_env import RouteAction, RouteEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with RouteEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(RouteAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/route_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Hackathon Validation Checklist

- OpenEnv compliance: `openenv validate --verbose`
- Docker build: `docker build -t route_env-env:latest -f server/Dockerfile .`
- Baseline script: `inference.py` is in project root and uses OpenAI client + required env vars.
- Multi-trajectory benchmark mode: set `NUM_TRAJECTORIES` and `MAX_STEPS_PER_TRAJECTORY`.

## Project Structure

```
route_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # RouteEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── route_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
