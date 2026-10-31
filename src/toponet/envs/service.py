"""Envs / service.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

import math
import numpy as np
from toponet.envs.traffic import BackgroundTraffic
from toponet.envs.allocation import allocate_cell


def integrate_service(
    traffic,
    duration_s,
    background_capacity_mbps,
    serving_cell=None,
    user_demand_mbps=20.0,
    user_capacity_fn=None,
    interruption_s=0.0,
):
    """Advance traffic and integrate service over each constant-demand interval.

    user_capacity_fn(current_demand_copy) returns the selected cell's full-resource
    Mbps, or zero if radio-ineligible. It must not change traffic or radio state.
    It can recompute current load-dependent interference at a fixed position.
    interruption_s is remaining interruption at interval start. No user resources
    are allocated during interruption; unmet user demand is still counted.
    Returns Mbit integrals, duration-weighted Mbps/fractions, and pending pause.
    No handover decision, reward, movement, or learning occurs here.
    """
    duration = float(duration_s)
    pause = float(interruption_s)
    request = float(user_demand_mbps)
    caps = np.asarray(background_capacity_mbps, dtype=float)
    n = traffic.num_cells
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid duration.")
    if not math.isfinite(pause) or pause < 0:
        raise ValueError("Invalid interruption.")
    if not math.isfinite(request) or request < 0:
        raise ValueError("Invalid user request.")
    if caps.shape != (n,) or not np.isfinite(caps).all() or (caps < 0).any():
        raise ValueError("Invalid background capacities.")
    if serving_cell is not None:
        if (
            isinstance(serving_cell, bool)
            or not isinstance(serving_cell, (int, np.integer))
            or not 0 <= serving_cell < n
        ):
            raise ValueError("Invalid serving cell.")
        if not callable(user_capacity_fn):
            raise ValueError("Provide a selected-cell capacity function.")
    start = traffic.time_s
    end = start + duration
    if end > traffic.duration_s + 1e-8:
        raise ValueError("Scenario does not cover interval.")
    end = min(end, traffic.duration_s)
    actual = end - start
    if actual <= 0:
        raise ValueError("Scenario finished.")
    pause_end = start + pause
    bg_requested = np.zeros(n)
    bg_delivered = np.zeros(n)
    bg_unmet = np.zeros(n)
    utilization = np.zeros(n)
    counts = np.zeros(n)
    user_delivered = 0.0
    segments = 0
    while traffic.time_s < end:
        now = traffic.time_s
        # Consume zero-time/boundary events using the existing accounting routine.
        # Normally advance() already consumed events at the preceding endpoint.
        index = traffic._event_index
        if index < len(traffic._events) and traffic._events[index][0] <= now:
            raise RuntimeError("Traffic has unconsumed events at the current time.")
        boundary = end
        if index < len(traffic._events):
            boundary = min(boundary, traffic._events[index][0])
        if now < pause_end < boundary:
            boundary = pause_end
        gap = boundary - now
        if gap <= 0:
            raise RuntimeError("Nonpositive integration interval.")
        demand = traffic.current_demand_mbps.copy()
        active_user = serving_cell is not None and now >= pause_end
        user_capacity = 0.0
        if active_user:
            user_capacity = float(user_capacity_fn(demand.copy()))
            if not math.isfinite(user_capacity) or user_capacity < 0:
                raise ValueError("Invalid user radio capacity.")
        for i in range(n):
            r = allocate_cell(
                float(demand[i]),
                float(caps[i]),
                request if active_user and i == serving_cell else 0.0,
                user_capacity if active_user and i == serving_cell else 0.0,
            )
            bg_delivered[i] += gap * r["background_delivered_mbps"]
            bg_unmet[i] += gap * r["background_unmet_mbps"]
            utilization[i] += gap * r["utilization"]
            user_delivered += gap * r["user_delivered_mbps"]
        bg_requested += gap * demand
        counts += gap * traffic.active_counts
        traffic.advance(gap)
        segments += 1
    user_requested = request * actual if serving_cell is not None else 0.0
    user_unmet = max(0.0, user_requested - user_delivered)
    return {
        "duration_s": actual,
        "segments": segments,
        "background_requested_mbit": bg_requested,
        "background_delivered_mbit": bg_delivered,
        "background_unmet_mbit": bg_unmet,
        "mean_background_delivered_mbps": bg_delivered / actual,
        "mean_background_demand_mbps": bg_requested / actual,
        "mean_active_counts": counts / actual,
        "mean_utilization": utilization / actual,
        "user_requested_mbit": user_requested,
        "user_delivered_mbit": user_delivered,
        "user_unmet_mbit": user_unmet,
        "mean_user_delivered_mbps": user_delivered / actual,
        "interruption_remaining_s": max(0.0, pause - actual),
    }


def controlled_traffic():
    t = BackgroundTraffic(
        [{"assumed_load_fraction": 0.0, "prb_capacity": 100}], 2.0, seed=7
    )
    t._events = [(0.5, 1, 0, 100.0), (1.0, -1, 0, 100.0)]
    return t


def self_test():
    # Demand is 0 for half a second, then 100 for half a second.
    # With 50 Mbps capacity: true delivered mean 25, NOT min(mean demand,50)=50.
    t = controlled_traffic()
    r = integrate_service(t, 1.0, [50.0])
    assert np.isclose(r["mean_background_delivered_mbps"][0], 25.0)
    assert np.isclose(r["background_unmet_mbit"][0], 25.0)
    assert np.isclose(r["mean_background_demand_mbps"][0], 50.0)
    assert t.active_counts[0] == 0 and t.current_demand_mbps[0] == 0
    assert t.total_arrivals == 1 and t.total_departures == 1
    assert np.allclose(
        r["background_requested_mbit"],
        r["background_delivered_mbit"] + r["background_unmet_mbit"],
    )
    # Pause first .25s; empty-cell user receives 20Mbps for .25s;
    # overloaded second half scales user to 20/2.5=8 Mbps.
    t = controlled_traffic()
    r = integrate_service(t, 1.0, [50.0], 0, 20.0, lambda d: 40.0, 0.25)
    assert np.isclose(r["user_delivered_mbit"], 9.0)
    assert np.isclose(r["user_requested_mbit"], 20.0) and np.isclose(
        r["user_unmet_mbit"], 11.0
    )
    assert np.all((r["mean_utilization"] >= 0) & (r["mean_utilization"] <= 1.0))
    # Partitioning with carried interruption must preserve service and events.
    a = controlled_traffic()
    b = controlled_traffic()
    whole = integrate_service(a, 1.0, [50.0], 0, 20.0, lambda d: 40.0, 0.7)
    first = integrate_service(b, 0.4, [50.0], 0, 20.0, lambda d: 40.0, 0.7)
    second = integrate_service(
        b, 0.6, [50.0], 0, 20.0, lambda d: 40.0, first["interruption_remaining_s"]
    )
    assert np.isclose(
        whole["user_delivered_mbit"],
        first["user_delivered_mbit"] + second["user_delivered_mbit"],
    )
    assert np.allclose(
        whole["background_delivered_mbit"],
        first["background_delivered_mbit"] + second["background_delivered_mbit"],
    )
    assert (
        a.total_arrivals == b.total_arrivals
        and a.total_departures == b.total_departures
    )
    z = integrate_service(controlled_traffic(), 0.5, [50.0], 0, 20.0, lambda d: 0.0)
    assert z["user_delivered_mbit"] == 0.0 and z["user_unmet_mbit"] == 10.0
    for kwargs in [
        {"duration_s": 0.0},
        {"interruption_s": -1.0},
        {"serving_cell": 1},
        {"background_capacity_mbps": [-1.0]},
    ]:
        params = dict(
            duration_s=0.5,
            background_capacity_mbps=[50.0],
            serving_cell=0,
            user_capacity_fn=lambda d: 40.0,
        )
        params.update(kwargs)
        try:
            integrate_service(controlled_traffic(), **params)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted.")
    print(
        "Interval tests passed: nonlinear allocation, event boundaries, interruption, split intervals, zero radio capacity and conservation."
    )
