import os
import sys
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

try:
    from route_env import RouteAction, RouteEnv
except ModuleNotFoundError:
    # Support running as: python inference.py from route_env/ directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from route_env import RouteAction, RouteEnv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
NUM_TRAJECTORIES = int(os.getenv("NUM_TRAJECTORIES", "1"))
MAX_STEPS_PER_TRAJECTORY = int(os.getenv("MAX_STEPS_PER_TRAJECTORY", "200"))
USE_OPENLLM_AGENT = os.getenv("USE_OPENLLM_AGENT", "false").lower() in (
    "1",
    "true",
    "yes",
)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
STRICT_SINGLE_EPISODE = os.getenv("STRICT_SINGLE_EPISODE", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Strict mode keeps exactly one [START] and one [END] per run for validators.

WAIT_FOR_SERVER_SECONDS = float(os.getenv("WAIT_FOR_SERVER_SECONDS", "30"))
WAIT_FOR_SERVER_POLL_SECONDS = float(os.getenv("WAIT_FOR_SERVER_POLL_SECONDS", "0.5"))

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def choose_action(observation: Any) -> tuple[RouteAction, str]:
    rides = observation.available_rides or []
    if rides:
        best = max(rides, key=lambda r: float(r.get("fare", 0.0)))
        action = RouteAction(action_type="accept_ride", ride_id=int(best["ride_id"]))
        return action, f"accept_ride({best['ride_id']})"

    demand = observation.live_demand_matrix or []
    current_node = int(observation.current_node)
    if demand:
        best_node = max(range(len(demand)), key=lambda idx: float(demand[idx]))
        if best_node != current_node:
            action = RouteAction(action_type="reposition", target_node=int(best_node))
            return action, f"reposition({best_node})"

    action = RouteAction(action_type="wait")
    return action, "wait()"


def choose_action_with_openllm(observation: Any) -> tuple[RouteAction, str]:
    rides = observation.available_rides or []
    demand = observation.live_demand_matrix or []
    payload = {
        "current_node": int(observation.current_node),
        "driver_status": str(observation.driver_status),
        "shift_hours_remaining": float(observation.shift_hours_remaining),
        "available_rides": rides,
        "live_demand_matrix": demand,
    }

    system_prompt = (
        "You are a dispatch RL policy. Return ONLY JSON with keys: "
        "action_type, ride_id, target_node. "
        "action_type must be one of wait|accept_ride|reposition."
    )
    user_prompt = (
        "Choose the best next action to maximize cumulative reward. "
        "If choosing accept_ride, use one ride_id from available_rides. "
        "If choosing reposition, pick likely-demand node from live_demand_matrix. "
        f"Observation: {json.dumps(payload)}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
        data = json.loads(raw)
        action_type = str(data.get("action_type", "wait"))
        ride_id = data.get("ride_id")
        target_node = data.get("target_node")

        if action_type == "accept_ride" and ride_id is not None:
            action = RouteAction(action_type="accept_ride", ride_id=int(ride_id))
            return action, f"accept_ride({int(ride_id)})"
        if action_type == "reposition" and target_node is not None:
            action = RouteAction(action_type="reposition", target_node=int(target_node))
            return action, f"reposition({int(target_node)})"
        return RouteAction(action_type="wait"), "wait()"
    except Exception:
        return choose_action(observation)


def run_trajectory(env: RouteEnv, trajectory_idx: int) -> tuple[bool, int, float]:
    model_name = MODEL_NAME
    rewards: list[str] = []
    success = False
    step_idx = 0
    cumulative_reward = 0.0
    task_label = f"trajectory_{trajectory_idx}"
    print(f"[START] task={task_label} env=route_env model={model_name}")

    try:
        with env.sync() as client:
            reset_result = client.reset()
            observation = reset_result.observation
            task_label = f"{observation.task_name}_traj{trajectory_idx}"

            while step_idx < MAX_STEPS_PER_TRAJECTORY:
                step_idx += 1
                if USE_OPENLLM_AGENT:
                    action, action_str = choose_action_with_openllm(observation)
                else:
                    action, action_str = choose_action(observation)
                result = client.step(action)
                observation = result.observation

                reward = 0.0 if result.reward is None else float(result.reward)
                cumulative_reward += reward
                rewards.append(f"{reward:.2f}")
                done = bool(result.done)
                error = observation.last_action_error if observation.last_action_error else "null"
                print(
                    f"[STEP] step={step_idx} action={action_str} reward={reward:.2f} "
                    f"done={'true' if done else 'false'} error={error}"
                )

                if done:
                    success = cumulative_reward >= 0.0 and (
                        observation.normalized_progress_score >= 0.25
                    )
                    break

        if not success and step_idx >= MAX_STEPS_PER_TRAJECTORY:
            success = cumulative_reward >= 0.0
    except Exception:
        success = False
    finally:
        print(
            f"[END] success={'true' if success else 'false'} steps={step_idx} "
            f"rewards={','.join(rewards)}"
        )

    return success, step_idx, cumulative_reward


def run_episode() -> None:
    trajectories = 1 if STRICT_SINGLE_EPISODE else max(1, NUM_TRAJECTORIES)
    # If inference starts immediately after the Docker container, the server
    # may still be initializing. Poll /health to avoid empty trajectories.
    deadline = time.time() + WAIT_FOR_SERVER_SECONDS
    while time.time() < deadline:
        try:
            with urlopen(f"{ENV_BASE_URL}/health", timeout=2) as resp:
                if getattr(resp, "status", 200) == 200:
                    break
        except (URLError, HTTPError):
            pass
        time.sleep(WAIT_FOR_SERVER_POLL_SECONDS)
    for trajectory_idx in range(1, trajectories + 1):
        env = RouteEnv(base_url=ENV_BASE_URL)
        run_trajectory(env, trajectory_idx)


if __name__ == "__main__":
    # Make a lightweight OpenAI-client request, but never fail the rollout if endpoint
    # disables this route.
    try:
        _ = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "healthcheck"}],
            max_tokens=2,
        )
    except Exception:
        pass
    run_episode()
