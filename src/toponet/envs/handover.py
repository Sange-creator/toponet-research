"""Envs / handover.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import math
import numpy as np
from toponet.data.towers import load_cells
from toponet.data.geometry import latlon_to_xy
from toponet.data.graph import build_neighbor_graph
from toponet.data.movement import RouteMotion
from toponet.envs.traffic import BackgroundTraffic
from toponet.envs.radio import RadioModel, RadioSettings
from toponet.envs.service import integrate_service


@dataclass(frozen=True)
class HandoverSettings:
    interruption_s: float = 0.1
    ping_pong_window_s: float = 3.0


class HandoverEnvironment:
    def __init__(
        self, cells, xy_m, neighbors, motion, settings=None, radio_settings=None
    ):
        self.cells = cells
        self.xy_m = np.array(xy_m, dtype=float, copy=True)
        self.n = len(cells)
        self.motion = motion
        self.neighbors = np.array(neighbors, dtype=bool, copy=True)
        self.settings = HandoverSettings() if settings is None else settings
        self.radio_settings = (
            RadioSettings() if radio_settings is None else radio_settings
        )
        if (
            self.neighbors.shape != (self.n, self.n)
            or not np.array_equal(self.neighbors, self.neighbors.T)
            or np.diag(self.neighbors).any()
        ):
            raise ValueError("Expected symmetric neighbor-only matrix.")
        if (
            not math.isfinite(self.settings.interruption_s)
            or self.settings.interruption_s < 0
            or not math.isfinite(self.settings.ping_pong_window_s)
            or self.settings.ping_pong_window_s <= 0
        ):
            raise ValueError("Invalid handover settings.")
        self.duration_s = motion.route_length_m / motion.speed_mps
        self.reset()

    def reset(self, traffic_seed=7, radio_seed=107, spatial_mode="nominal"):
        self.motion.reset()
        self.traffic = BackgroundTraffic(
            self.cells, self.duration_s, traffic_seed, spatial_mode=spatial_mode
        )
        self.radio = RadioModel(self.cells, self.xy_m, radio_seed, self.radio_settings)
        initial = self.radio.snapshot(
            self.motion.position_m, self.traffic.current_demand_mbps
        )
        # Strongest signal also maximizes eligibility; if all weak, start strongest.
        self.serving_cell = int(np.argmax(initial["rsrp_dbm"]))
        self.initial_serving_cell = self.serving_cell
        self.previous_cell = None
        self.last_handover_time_s = None
        self.interruption_remaining_s = 0.0
        self.total_handovers = 0
        self.ping_pongs = 0
        self.weak_duration_s = 0.0
        self.weak_episodes = 0
        self._previous_interval_weak = False
        self.user_delivered_mbit = 0.0
        self.user_requested_mbit = 0.0
        self.background_delivered_mbit = np.zeros(self.n)
        self.background_requested_mbit = np.zeros(self.n)
        self.utilization_integral = np.zeros(self.n)
        self.elapsed_s = 0.0
        self.handover_events = []
        return self.observe()

    def observe(self):
        s = self.radio.snapshot(
            self.motion.position_m, self.traffic.current_demand_mbps
        )
        mask = self.neighbors[self.serving_cell] & s["radio_eligible"]
        # Staying is always an executable fallback, including weak-signal periods.
        mask[self.serving_cell] = True
        if self.interruption_remaining_s > 1e-12:
            mask[:] = False
            mask[self.serving_cell] = True
        return {
            "time_s": self.motion.time_s,
            "position_m": self.motion.position_m.copy(),
            "lookahead_position_m": self.motion.lookahead_position_m().copy(),
            "serving_cell": self.serving_cell,
            "previous_cell": self.previous_cell,
            "time_since_handover_s": None
            if self.last_handover_time_s is None
            else self.motion.time_s - self.last_handover_time_s,
            "interruption_remaining_s": self.interruption_remaining_s,
            "action_mask": mask.copy(),
            "rsrp_dbm": s["rsrp_dbm"].copy(),
            "sinr_db": s["sinr_db"].copy(),
            "candidate_user_rate_mbps": s["candidate_user_rate_mbps"].copy(),
            "background_demand_mbps": self.traffic.current_demand_mbps.copy(),
            "active_background_sessions": self.traffic.active_counts.copy(),
            "background_reference_utilization": s["background_utilization"].copy(),
        }

    def step(self, action):
        if self.motion.done:
            raise RuntimeError("Episode finished. Reset first.")
        obs = self.observe()
        if (
            isinstance(action, bool)
            or not isinstance(action, (int, np.integer))
            or not 0 <= action < self.n
            or not obs["action_mask"][action]
        ):
            raise ValueError("Action is not eligible.")
        action = int(action)
        now = self.motion.time_s
        old = self.serving_cell
        switched = action != old
        ping_pong = False
        if switched:
            ping_pong = (
                self.previous_cell == action
                and self.last_handover_time_s is not None
                and now - self.last_handover_time_s < self.settings.ping_pong_window_s
            )
            self.total_handovers += 1
            self.ping_pongs += int(ping_pong)
            self.previous_cell = old
            self.last_handover_time_s = now
            self.serving_cell = action
            self.interruption_remaining_s = self.settings.interruption_s
            self.handover_events.append(
                {
                    "time_s": now,
                    "distance_m": self.motion.distance_m,
                    "from_node": old,
                    "to_node": action,
                    "from_site_id": self.cells[old]["site_id"],
                    "to_site_id": self.cells[action]["site_id"],
                    "ping_pong": bool(ping_pong),
                }
            )
        position = self.motion.position_m.copy()
        weak = bool(
            obs["rsrp_dbm"][action] < self.radio.settings.minimum_target_rsrp_dbm
        )
        if weak and not self._previous_interval_weak:
            self.weak_episodes += 1
        self._previous_interval_weak = weak
        duration = min(
            self.motion.dt_s,
            (self.motion.route_length_m - self.motion.distance_m)
            / self.motion.speed_mps,
        )

        def capacity(demand):
            s = self.radio.snapshot(position, demand)
            return (
                s["user_full_resource_rate_mbps"][action]
                if s["radio_eligible"][action]
                else 0.0
            )

        service = integrate_service(
            self.traffic,
            duration,
            self.radio.background_capacity_mbps,
            action,
            self.radio.settings.user_demand_mbps,
            capacity,
            self.interruption_remaining_s,
        )
        movement = self.motion.step()
        self.radio.advance_shadowing(movement["distance_travelled_m"])
        if not np.isclose(self.traffic.time_s, self.motion.time_s, atol=1e-8, rtol=0):
            raise RuntimeError("Traffic/motion clocks diverged.")
        self.interruption_remaining_s = service["interruption_remaining_s"]
        self.elapsed_s += service["duration_s"]
        self.weak_duration_s += service["duration_s"] * weak
        self.user_delivered_mbit += service["user_delivered_mbit"]
        self.user_requested_mbit += service["user_requested_mbit"]
        self.background_delivered_mbit += service["background_delivered_mbit"]
        self.background_requested_mbit += service["background_requested_mbit"]
        self.utilization_integral += service["mean_utilization"] * service["duration_s"]
        info = {
            "decision_time_s": now,
            "end_time_s": self.motion.time_s,
            "decision_distance_m": movement["start_distance_m"],
            "end_distance_m": self.motion.distance_m,
            "serving_node": action,
            "serving_site_id": self.cells[action]["site_id"],
            "handover": switched,
            "ping_pong": bool(ping_pong),
            "weak_signal": weak,
            "serving_rsrp_dbm": float(obs["rsrp_dbm"][action]),
            "duration_s": service["duration_s"],
            "user_delivered_mbps": service["mean_user_delivered_mbps"],
            "background_delivered_mbps": float(
                service["mean_background_delivered_mbps"].sum()
            ),
            "traffic_subintervals": service["segments"],
        }
        return self.observe(), self.motion.done, info

    def summary(self):
        d = self.elapsed_s
        return {
            "elapsed_s": d,
            "initial_serving_site_id": self.cells[self.initial_serving_cell]["site_id"],
            "mean_user_delivered_mbps": self.user_delivered_mbit / d if d else None,
            "mean_background_delivered_mbps": float(
                self.background_delivered_mbit.sum() / d
            )
            if d
            else None,
            "user_delivered_mbit": self.user_delivered_mbit,
            "user_unmet_mbit": max(
                0.0, self.user_requested_mbit - self.user_delivered_mbit
            ),
            "background_delivered_mbit": self.background_delivered_mbit.tolist(),
            "background_unmet_mbit": np.maximum(
                0.0, self.background_requested_mbit - self.background_delivered_mbit
            ).tolist(),
            "mean_cell_utilization": (self.utilization_integral / d).tolist()
            if d
            else None,
            "handovers": self.total_handovers,
            "ping_pongs": self.ping_pongs,
            "weak_signal_duration_s": self.weak_duration_s,
            "weak_signal_episodes": self.weak_episodes,
        }


def self_test():
    cells = [
        {
            "site_id": i + 1,
            "assumed_load_fraction": 0.3,
            "prb_capacity": 100,
            "downlink_frequency_mhz": 1815.0,
            "tx_power_dbm": 43.0,
            "height_m": 30.0,
        }
        for i in range(3)
    ]
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [1000.0, 0.0]])
    neighbors = build_neighbor_graph(xy, 200.0)[0]

    def env(length=100.0):
        return HandoverEnvironment(
            cells,
            xy,
            neighbors,
            RouteMotion([[0.0, 0.0], [length, 0.0]], [0.0, length]),
            radio_settings=RadioSettings(shadowing_std_db=0.0),
        )

    e = env()
    assert e.serving_cell == 0 and not e.observe()["action_mask"][2]
    before = e.traffic.time_s
    try:
        e.step(2)
    except ValueError:
        pass
    else:
        raise AssertionError("Ineligible action accepted.")
    assert e.traffic.time_s == before
    e.step(1)
    _, _, r = e.step(0)
    assert r["ping_pong"] and e.ping_pongs == 1 and e.total_handovers == 2
    e = env()
    e.step(1)
    for _ in range(5):
        e.step(1)
    e.step(0)
    assert e.ping_pongs == 0  # return at exactly 3s, window is strict <3.
    a = env()
    b = env()
    for _ in range(4):
        _, _, ra = a.step(0)
        _, _, rb = b.step(0)
        assert ra == rb
    assert np.array_equal(a.traffic.current_demand_mbps, b.traffic.current_demand_mbps)
    a.reset()
    assert a.elapsed_s == 0 and a.total_handovers == 0
    short = env(2.0)
    _, done, r = short.step(0)
    assert (
        done and np.isclose(r["duration_s"], 0.2) and np.isclose(short.elapsed_s, 0.2)
    )
    assert short.summary()["mean_user_delivered_mbps"] <= 20.0
    try:
        short.step(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Terminal step accepted.")
    # A deliberately weak radio must retain the stay fallback and count a run once.
    weak = env()
    weak.radio.shadowing_db[:] = -200.0
    assert weak.observe()["action_mask"].sum() == 1
    weak.step(0)
    weak.step(0)
    assert (
        weak.weak_episodes == 1
        and weak.weak_duration_s == 1.0
        and weak.user_delivered_mbit == 0.0
    )
    # Longer interruption carries over; it is execution, not a stability heuristic.
    p = env()
    p.settings = HandoverSettings(interruption_s=0.8)
    p.step(1)
    assert np.isclose(p.interruption_remaining_s, 0.3)
    assert p.observe()["action_mask"].sum() == 1
    p.step(1)
    assert p.interruption_remaining_s == 0.0
    print(
        "Environment tests passed: masks, initial association, rapid/delayed returns, seeds, reset, partial final step, weak runs and interruption carryover."
    )
