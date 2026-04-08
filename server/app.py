# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Route Env Environment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import math
import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from fastapi import HTTPException
from pydantic import BaseModel

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

# Flat structure — models and server files are all at repo root
from models import RouteAction, RouteObservation
from server.route_env_environment import RouteEnvironment
from tasks import TASKS, TASK_ORDER
from grader import score_episode

app = create_app(
    RouteEnvironment,
    RouteAction,
    RouteObservation,
    env_name="route_env",
    max_concurrent_envs=1,
)


# ── /tasks endpoint ───────────────────────────────────────────────────────────
# The OpenEnv validator calls GET /tasks to discover available tasks and
# confirm that at least 3 are registered.

@app.get("/tasks")
def list_tasks():
    """Return all registered tasks with their configurations."""
    return {
        "tasks": [
            {
                "name": name,
                "horizon_steps": cfg.horizon_steps,
                "node_count": cfg.node_count,
                "max_shift_hours": cfg.max_shift_hours,
                "has_grader": True,
            }
            for name, cfg in TASKS.items()
        ]
    }


# ── /grader endpoint ──────────────────────────────────────────────────────────
# The OpenEnv validator calls POST /grader to verify that scores are strictly
# in (0, 1) for each task.

class GraderRequest(BaseModel):
    task_name: str
    step_count: int = 0
    completed_rides: int = 0
    late_rides: int = 0
    total_reward: float = 0.0


@app.post("/grader")
def grade(request: GraderRequest):
    """Score one episode. Returns a float strictly in (0.0, 1.0)."""
    if request.task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_name}'. Valid tasks: {TASK_ORDER}",
        )
    score = score_episode(
        step_count=request.step_count,
        completed_rides=request.completed_rides,
        late_rides=request.late_rides,
        total_reward=request.total_reward,
    )
    # Belt-and-suspenders: guarantee strictly (0, 1) before returning
    assert 0.0 < score < 1.0, f"Grader returned out-of-range score: {score}"
    return {
        "task_name": request.task_name,
        "score": score,
    }


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()