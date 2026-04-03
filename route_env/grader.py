"""Deterministic grader utilities returning scores in [0, 1]."""


def score_episode(
    step_count: int,
    completed_rides: int,
    late_rides: int,
    total_reward: float,
) -> float:
    """Score one trajectory using efficiency, punctuality, and reward quality."""
    denominator = max(1, step_count)
    ride_efficiency = completed_rides / denominator
    punctuality = 1.0 - (late_rides / max(1, completed_rides))
    reward_quality = (total_reward / max(1.0, float(step_count))) + 0.5
    normalized = (
        0.45 * ride_efficiency
        + 0.35 * max(0.0, punctuality)
        + 0.20 * max(0.0, reward_quality)
    )
    return float(min(1.0, max(0.0, normalized)))
