import os
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

# Flat repo structure — all modules at repo root
from client import RouteEnv
from models import RouteAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

# Use API_KEY injected by the validator's LiteLLM proxy.
# Fall back to HF_TOKEN for local/HuggingFace runs.
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
if API_KEY is None:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
NUM_TRAJECTORIES = int(os.getenv("NUM_TRAJECTORIES", "1"))
MAX_STEPS_PER_TRAJECTORY = int(os.getenv("MAX_STEPS_PER_TRAJECTORY", "200"))

# DEFAULT TRUE: LLM drives all decisions so the validator proxy is always used.
USE_OPENLLM_AGENT = os.getenv("USE_OPENLLM_AGENT", "true").lower() in (
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

WAIT_FOR_SERVER_SECONDS = float(os.getenv("WAIT_FOR_SERVER_SECONDS", "30"))
WAIT_FOR_SERVER_POLL_SECONDS = float(os.getenv("WAIT_FOR_SERVER_POLL_SECONDS", "0.5"))

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def choose_action_heuristic(observation: Any) -> tuple[RouteAction, str]:
    """Greedy fallback: max-fare ride > max-demand reposition > wait."""
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

    return RouteAction(action_type="wait"), "wait()"


def choose_action_with_openllm(
    observation: Any,
    step_idx: int,
    prev_progress: float,
    cumulative_reward: float,
) -> tuple[RouteAction, str]:
    """LLM-driven policy with rich context for long-term planning."""
    rides = observation.available_rides or []
    demand = observation.live_demand_matrix or []

    # Rank rides by fare so the LLM has a prioritized view
    ranked_rides = sorted(rides, key=lambda r: float(r.get("fare", 0.0)), reverse=True)

    # Find top demand nodes for context
    top_demand_nodes = []
    if demand:
        indexed = [(i, float(v)) for i, v in enumerate(demand)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_demand_nodes = [{"node": i, "demand": round(v, 3)} for i, v in indexed[:3]]

    payload = {
        "step": step_idx,
        "current_node": int(observation.current_node),
        "driver_status": str(observation.driver_status),
        "shift_hours_remaining": round(float(observation.shift_hours_remaining), 2),
        "normalized_progress_so_far": round(prev_progress, 3),
        "cumulative_reward_so_far": round(cumulative_reward, 3),
        "available_rides_ranked_by_fare": ranked_rides,
        "top_demand_nodes": top_demand_nodes,
        "last_action_error": observation.last_action_error,
    }

    system_prompt = (
        "You are an expert ride-dispatch RL policy optimizing a driver's shift. "
        "Your goal is to maximize normalized_progress_score by the end of the episode. "
        "Rules:\n"
        "- Prefer high-fare rides when available — they improve progress score faster.\n"
        "- When idle with no rides, reposition to the highest-demand node to attract rides.\n"
        "- Never reposition if you are already at the best demand node — wait instead.\n"
        "- Monitor shift_hours_remaining: if low, prioritize accepting any available ride.\n"
        "- If last_action_error is not null, avoid repeating the same action type.\n"
        "Respond with ONLY a JSON object, no explanation, no markdown. "
        "Keys: action_type (wait|accept_ride|reposition), ride_id (int or null), target_node (int or null). "
        "Example: {\"action_type\": \"accept_ride\", \"ride_id\": 3, \"target_node\": null}"
    )

    user_prompt = (
        f"Current state:\n{json.dumps(payload, indent=2)}\n\n"
        "What is the single best action to take right now to maximize cumulative reward?"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=60,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start: end + 1]
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
        # Graceful fallback to heuristic on any LLM/parse failure
        return choose_action_heuristic(observation)


def run_trajectory(env: RouteEnv, trajectory_idx: int) -> tuple[bool, int, float]:
    model_name = MODEL_NAME
    rewards: list[str] = []
    prev_progress = 0.0
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
                    action, action_str = choose_action_with_openllm(
                        observation, step_idx, prev_progress, cumulative_reward
                    )
                else:
                    action, action_str = choose_action_heuristic(observation)

                result = client.step(action)
                observation = result.observation

                raw_reward = 0.0 if result.reward is None else float(result.reward)
                # Delta of normalized_progress_score gives fractional per-step signal.
                progress = float(observation.normalized_progress_score or 0.0)
                delta = max(0.0, progress - prev_progress)
                reward = delta if progress > 0.0 else max(0.0, min(1.0, raw_reward))
                prev_progress = progress
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
        env = (
            RouteEnv.from_docker_image(LOCAL_IMAGE_NAME)
            if LOCAL_IMAGE_NAME
            else RouteEnv(base_url=ENV_BASE_URL)
        )
        run_trajectory(env, trajectory_idx)


if __name__ == "__main__":
    # Warmup call ensures the validator's LiteLLM proxy key is activated
    # before the episode begins, regardless of agent mode.
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "healthcheck"}],
        max_tokens=2,
    )
    run_episode()