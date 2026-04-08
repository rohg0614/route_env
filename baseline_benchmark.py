"""Deterministic baseline benchmark across easy/medium/hard tasks."""

import os
import sys
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from pathlib import Path
from statistics import mean

try:
    from route_env import RouteAction, RouteEnv
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from route_env import RouteAction, RouteEnv


def choose_heuristic_action(observation) -> RouteAction:
    rides = observation.available_rides or []
    if rides:
        best = max(rides, key=lambda r: float(r.get("fare", 0.0)))
        return RouteAction(action_type="accept_ride", ride_id=int(best["ride_id"]))

    demand = observation.live_demand_matrix or []
    current_node = int(observation.current_node)
    if demand:
        best_node = max(range(len(demand)), key=lambda idx: float(demand[idx]))
        if best_node != current_node:
            return RouteAction(action_type="reposition", target_node=int(best_node))
    return RouteAction(action_type="wait")


def run_task(task_name: str, episodes: int = 3, max_steps: int = 120) -> float:
    scores = []
    env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
    wait_seconds = float(os.getenv("WAIT_FOR_SERVER_SECONDS", "30"))
    poll_seconds = float(os.getenv("WAIT_FOR_SERVER_POLL_SECONDS", "0.5"))
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            with urlopen(f"{env_url}/health", timeout=2) as resp:
                if getattr(resp, "status", 200) == 200:
                    break
        except (URLError, HTTPError):
            pass
        time.sleep(poll_seconds)
    with RouteEnv(base_url=env_url).sync() as env:
        for ep in range(episodes):
            result = env.reset(task_name=task_name, seed=1000 + ep)
            obs = result.observation
            steps = 0
            while steps < max_steps and not obs.done:
                steps += 1
                step_result = env.step(choose_heuristic_action(obs))
                obs = step_result.observation
            scores.append(float(obs.normalized_progress_score))
    return mean(scores)


def main() -> None:
    task_scores = {task: run_task(task) for task in ("easy", "medium", "hard")}
    overall = mean(task_scores.values())
    print("Baseline benchmark (0.0 - 1.0):")
    for task, score in task_scores.items():
        print(f"  {task}: {score:.4f}")
    print(f"  overall: {overall:.4f}")
    artifact = {"task_scores": task_scores, "overall": overall}
    with open("baseline_scores.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print("  wrote baseline_scores.json")


if __name__ == "__main__":
    main()
