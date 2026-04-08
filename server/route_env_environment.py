# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Route dispatch optimization environment implementation."""

import math
import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Flat structure — all modules are at repo root
from models import RouteAction, RouteObservation
from tasks import TASKS, TASK_ORDER, TaskConfig
from grader import score_episode


class RouteEnvironment(Environment):
    """Graph-based stochastic dispatch environment with dense rewards."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()
        self._task_idx = -1
        self._tasks = [TASKS[name] for name in TASK_ORDER]
        self._base_seed = int(os.getenv("SEED", "42"))
        self._max_fare = 60.0
        self._max_distance = 20.0
        self._reset_internal()

    def _reset_internal(self) -> None:
        self._task = self._tasks[max(self._task_idx, 0)]
        self._driver_node = 0
        self._driver_status = "idle"
        self._sim_hour = 6.0
        self._shift_hours_remaining = self._task.max_shift_hours
        self._rides: list[dict] = []
        self._next_ride_id = 1
        self._idle_wait_steps = 0
        self._total_reward = 0.0
        self._completed_rides = 0
        self._late_rides = 0
        self._total_fare = 0.0
        self._empty_distance = 0.0
        self._build_graph()
        self._spawn_rides()

    def _build_graph(self) -> None:
        n = self._task.node_count
        self._adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
        self._edge_distance: dict[tuple[int, int], float] = {}
        for i in range(n):
            right = (i + 1) % n
            left = (i - 1) % n
            for j in (right, left):
                if j not in self._adjacency[i]:
                    self._adjacency[i].append(j)
                d = 2.0 + abs(i - j) + self._rng.random() * 2.0
                self._edge_distance[(i, j)] = d
                self._edge_distance[(j, i)] = d

    def _poisson(self, lam: float) -> int:
        k = 0
        p = 1.0
        threshold = math.exp(-lam)
        while p > threshold:
            k += 1
            p *= self._rng.random()
        return max(0, k - 1)

    def _node_intensity(self, node: int) -> float:
        peak = 1.0 + 0.6 * math.sin((self._sim_hour - 8.0) * math.pi / 12.0)
        locality = 0.8 + (node / max(1, self._task.node_count - 1)) * 0.5
        return max(0.1, self._task.base_lambda * peak * locality)

    def _spawn_rides(self) -> None:
        for node in range(self._task.node_count):
            arrivals = self._poisson(self._node_intensity(node))
            for _ in range(arrivals):
                destination = self._rng.randrange(0, self._task.node_count)
                while destination == node:
                    destination = self._rng.randrange(0, self._task.node_count)
                estimated_distance = self._task.distance_scale * (
                    0.25 + abs(destination - node) / self._task.node_count
                )
                fare = min(self._max_fare, 7.0 + estimated_distance * (1.3 + self._rng.random()))
                wait_time = self._rng.randint(0, 8)
                self._rides.append(
                    {
                        "ride_id": self._next_ride_id,
                        "origin": node,
                        "destination": destination,
                        "fare": round(fare, 2),
                        "distance": round(estimated_distance, 2),
                        "wait_time": wait_time,
                    }
                )
                self._next_ride_id += 1

    def _live_demand(self) -> list[float]:
        counts = [0 for _ in range(self._task.node_count)]
        for ride in self._rides:
            counts[ride["origin"]] += 1
        total = max(1, sum(counts))
        return [round(c / total, 4) for c in counts]

    def _rides_at_current_node(self) -> list[dict]:
        return [r for r in self._rides if r["origin"] == self._driver_node][:8]

    def _advance_time(self, step_hours: float = 5 / 60) -> None:
        self._sim_hour = (self._sim_hour + step_hours) % 24.0
        self._shift_hours_remaining = max(0.0, self._shift_hours_remaining - step_hours)
        self._state.step_count += 1
        self._spawn_rides()

    def _grader_score(self) -> float:
        return score_episode(
            step_count=self._state.step_count,
            completed_rides=self._completed_rides,
            late_rides=self._late_rides,
            total_reward=self._total_reward,
        )

    def _build_observation(
        self, reward: float, done: bool, last_action_error: str | None
    ) -> RouteObservation:
        theta = 2 * math.pi * (self._sim_hour / 24.0)
        return RouteObservation(
            task_name=self._task.name,
            current_node=self._driver_node,
            time_of_day_sin=round(math.sin(theta), 6),
            time_of_day_cos=round(math.cos(theta), 6),
            driver_status=self._driver_status,
            shift_hours_remaining=round(self._shift_hours_remaining, 3),
            live_demand_matrix=self._live_demand(),
            available_rides=self._rides_at_current_node(),
            last_action_error=last_action_error,
            normalized_progress_score=round(self._grader_score(), 4),
            done=done,
            reward=round(reward, 4),
            metadata={
                "task_horizon": self._task.horizon_steps,
                "completed_rides": self._completed_rides,
                "late_rides": self._late_rides,
                "total_fare": round(self._total_fare, 2),
                "empty_distance": round(self._empty_distance, 2),
                "grader_score_0_to_1": round(self._grader_score(), 4),
            },
        )

    def reset(
        self,
        task_name: str | None = None,
        seed: int | None = None,
    ) -> RouteObservation:
        # ... [existing code] ...
        self._reset_internal()
        # CHANGE: Update 0.0 to 0.01
        return self._build_observation(reward=0.01, done=False, last_action_error=None)

    def step(self, action: RouteAction) -> RouteObservation:
        reward = 0.0
        last_action_error: str | None = None
        did_reposition = 0.0
        empty_distance = 0.0
        fare = 0.0
        waiting_time = float(self._idle_wait_steps)
        late_penalty = 0.0
        completed_ride = 0.0

        if self._shift_hours_remaining <= 0:
            obs = self._build_observation(
                reward=-1.0,
                done=True,
                last_action_error="shift_exhausted",
            )
            self._total_reward += 0.01
            return obs

        if action.action_type == "wait":
            self._driver_status = "idle"
            self._idle_wait_steps += 1

        elif action.action_type == "reposition":
            if action.target_node is None:
                last_action_error = "missing_target_node"
                reward -= 0.5
            elif action.target_node not in self._adjacency.get(self._driver_node, []):
                last_action_error = "target_not_adjacent"
                reward -= 0.6
            else:
                did_reposition = 1.0
                self._driver_status = "en_route"
                empty_distance = self._edge_distance[(self._driver_node, action.target_node)]
                self._empty_distance += empty_distance
                self._driver_node = action.target_node
                self._idle_wait_steps += 1

        elif action.action_type == "accept_ride":
            if action.ride_id is None:
                last_action_error = "missing_ride_id"
                reward -= 0.5
            else:
                selected = None
                for ride in self._rides:
                    if ride["ride_id"] == action.ride_id and ride["origin"] == self._driver_node:
                        selected = ride
                        break
                if selected is None:
                    last_action_error = "ride_not_available_at_node"
                    reward -= 0.6
                else:
                    self._rides.remove(selected)
                    completed_ride = 1.0
                    self._driver_status = "busy"
                    fare = float(selected["fare"])
                    waiting_time = float(selected["wait_time"])
                    if waiting_time > (self._task.lateness_budget * 10.0):
                        late_penalty = 1.0
                        self._late_rides += 1
                    self._driver_node = int(selected["destination"])
                    self._completed_rides += 1
                    self._total_fare += fare
                    self._idle_wait_steps = 0
        else:
            last_action_error = "unsupported_action_type"
            reward -= 0.7

        reward += 1.0 * (fare / self._max_fare)
        reward -= 0.9 * (empty_distance / self._max_distance)
        reward -= 0.15 * did_reposition
        if completed_ride == 1.0:
            reward += 2.0 * math.exp(-0.1 * waiting_time)
        reward -= 1.5 * late_penalty
        reward = max(0.01, min(0.99, reward))
        self._advance_time()
        done = (
            self._state.step_count >= self._task.horizon_steps
            or self._shift_hours_remaining <= 0.0
        )
        self._total_reward += reward
        return self._build_observation(
            reward=reward,
            done=done,
            last_action_error=last_action_error,
        )

    @property
    def state(self) -> State:
        return self._state