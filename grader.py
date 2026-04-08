"""Deterministic grader utilities returning scores strictly in (0.0, 1.0)."""


def score_episode(
    step_count: int,
    completed_rides: int,
    late_rides: int,
    total_reward: float,
) -> float:
    """
    Score one trajectory. Output is STRICTLY between 0.0 and 1.0 (exclusive).

    Components:
    - time_comp:  slow linear crawl — ensures score rises even on idle episodes
    - ride_comp:  jump per completed ride — primary performance signal
    - fare_comp:  smoothed reward contribution — adds organic variation
    - late_penalty: deduction per late ride

    The starting buffer (0.05) and per-step base (0.004) guarantee the score
    is always above 0.0 from step 1. The hard clamp to (0.02, 0.98) ensures
    the validator's strict (0, 1) check always passes.
    """
    # 1. Base progression — 0.004 per step so a 60-step episode contributes 0.24
    time_comp = step_count * 0.004

    # 2. Performance jumps — 0.025 per completed ride
    ride_comp = completed_rides * 0.025

    # 3. Fare variation — log-scaled so large fares don't dominate
    import math
    fare_comp = math.log1p(max(0.0, total_reward)) * 0.015

    # 4. Late ride penalty
    late_deduction = late_rides * 0.012

    # 5. Aggregate with starting buffer
    raw = 0.06 + time_comp + ride_comp + fare_comp - late_deduction

    # 6. Strict clamp — NEVER touches 0.0 or 1.0
    final_score = max(0.02, min(0.98, raw))

    return round(float(final_score), 4)