"""Deterministic grader utilities returning varied scores in [0.01, 0.99]."""

def score_episode(
    step_count: int,
    completed_rides: int,
    late_rides: int,
    total_reward: float,
) -> float:
    """Score trajectory with dynamic, varied increments."""
    
    # 1. Base Progression (0.005 per step)
    # This provides a 'slow crawl' that leaves room for performance jumps.
    time_comp = step_count * 0.005
    
    # 2. Performance Jumps (0.02 per ride)
    # These create the '0.02' and '0.03' rewards you want to see.
    ride_comp = completed_rides * 0.02
    
    # 3. Fare Scaling (0.01 per unit of fare)
    # Adds the 'noise' that makes the rewards look organic.
    fare_comp = total_reward * 0.01
    
    # 4. Aggregate with a 0.05 starting buffer
    raw = 0.05 + time_comp + ride_comp + fare_comp - (late_rides * 0.01)
    
    # 5. Strict Clamp
    final_score = max(0.01, min(0.99, raw))
    
    return round(float(final_score), 4)