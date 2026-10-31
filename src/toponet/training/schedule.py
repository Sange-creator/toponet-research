"""Training / schedule.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

import math
from toponet.training.sampler import BalancedRouteSampler


def plan_complete_journeys(rows, train_ids, episodes, route_seed, speed_mps, dt_s):
    if type(episodes) != int or not 1 <= episodes <= 10000:
        raise ValueError("Episodes must be 1..10000")
    if (
        not math.isfinite(speed_mps)
        or not math.isfinite(dt_s)
        or speed_mps <= 0
        or dt_s <= 0
    ):
        raise ValueError("Invalid motion scales")
    sampler = BalancedRouteSampler([rows[i] for i in train_ids], route_seed)
    journeys = []
    total = 0
    for episode in range(1, episodes + 1):
        rid = sampler.next_route_id()
        length = float(rows[rid]["length_m"])
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Invalid route length")
        steps = math.ceil(length / (speed_mps * dt_s))
        total += steps
        journeys.append(
            dict(
                episode=episode,
                route=rid,
                steps=steps,
                cumulative_steps=total,
                length_m=length,
            )
        )
    return dict(
        requested_complete_episodes=episodes,
        total_environment_steps=total,
        route_sampler_seed=route_seed,
        speed_mps=speed_mps,
        dt_s=dt_s,
        journeys=journeys,
    )
