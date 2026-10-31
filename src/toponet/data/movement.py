"""Data / movement.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from pathlib import Path
import csv
import json
import math
import numpy as np


class RouteMotion:
    """Validate once, then interpolate positions by distance along the road."""

    def __init__(self, xy_m, cumulative_m, speed_mps=10.0, dt_s=0.5):
        self.xy_m = np.array(xy_m, dtype=float, copy=True)
        self.cumulative_m = np.array(cumulative_m, dtype=float, copy=True)
        self.speed_mps = float(speed_mps)
        self.dt_s = float(dt_s)
        if not math.isfinite(self.speed_mps) or self.speed_mps <= 0:
            raise ValueError("Speed must be positive and finite.")
        if not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("Step duration must be positive and finite.")
        if (
            not math.isfinite(self.speed_mps * self.dt_s)
            or self.speed_mps * self.dt_s <= 0
        ):
            raise ValueError("Speed times step duration must be finite and positive.")
        if self.xy_m.ndim != 2 or self.xy_m.shape[1] != 2 or len(self.xy_m) < 2:
            raise ValueError("Route positions must have shape [points, 2].")
        if self.cumulative_m.shape != (len(self.xy_m),):
            raise ValueError("Route positions and distance arrays do not match.")
        if not np.isfinite(self.xy_m).all() or not np.isfinite(self.cumulative_m).all():
            raise ValueError("Route coordinates and distances must be finite.")
        if abs(self.cumulative_m[0]) > 1e-8 or (np.diff(self.cumulative_m) <= 0).any():
            raise ValueError(
                "Route distances must start at zero and strictly increase."
            )
        lengths = np.linalg.norm(np.diff(self.xy_m, axis=0), axis=1)
        if not np.allclose(np.diff(self.cumulative_m), lengths, atol=1e-6, rtol=1e-8):
            raise ValueError("Route distances do not match road geometry.")
        self.route_length_m = float(self.cumulative_m[-1])
        self.xy_m.setflags(write=False)
        self.cumulative_m.setflags(write=False)
        self.reset()

    @classmethod
    def from_file(cls, filename, speed_mps=10.0, dt_s=0.5):
        with np.load(filename, allow_pickle=False) as data:
            return cls(data["xy_m"], data["cumulative_distance_m"], speed_mps, dt_s)

    def _position_at(self, distance_m):
        d = float(np.clip(distance_m, 0.0, self.route_length_m))
        i = min(
            int(np.searchsorted(self.cumulative_m, d, side="right")) - 1,
            len(self.xy_m) - 2,
        )
        weight = (d - self.cumulative_m[i]) / (
            self.cumulative_m[i + 1] - self.cumulative_m[i]
        )
        return self.xy_m[i] + weight * (self.xy_m[i + 1] - self.xy_m[i])

    @property
    def position_m(self):
        return self._position_at(self.distance_m)

    def lookahead_position_m(self):
        """One step ahead along the known route; not future traffic information."""
        return self._position_at(self.distance_m + self.speed_mps * self.dt_s)

    def reset(self):
        self.time_s = 0.0
        self.distance_m = 0.0
        self.step_count = 0
        self.done = False
        return self.position_m

    def step(self):
        if self.done:
            raise RuntimeError("Journey finished. Reset before taking another step.")
        old_time = self.time_s
        old_distance = self.distance_m
        old_position = self.position_m
        remaining_m = self.route_length_m - old_distance
        duration_s = min(self.dt_s, remaining_m / self.speed_mps)
        # Use a shorter final interval; never travel past the end of the road route.
        if remaining_m <= self.speed_mps * self.dt_s:
            self.distance_m = self.route_length_m
        else:
            self.distance_m = old_distance + self.speed_mps * duration_s
        self.time_s = self.distance_m / self.speed_mps
        self.step_count += 1
        self.done = self.distance_m >= self.route_length_m
        return {
            "decision_time_s": old_time,
            "end_time_s": self.time_s,
            "duration_s": duration_s,
            "start_distance_m": old_distance,
            "end_distance_m": self.distance_m,
            "start_position_m": old_position,
            "end_position_m": self.position_m,
            "distance_travelled_m": self.distance_m - old_distance,
            "done": self.done,
        }


def self_test():
    m = RouteMotion(
        [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]], [0.0, 3.0, 7.0], speed_mps=2.0, dt_s=2.0
    )
    assert np.allclose(m.position_m, [0.0, 0.0])
    assert np.allclose(m.lookahead_position_m(), [3.0, 1.0])
    a = m.step()
    assert np.allclose(a["end_position_m"], [3.0, 1.0]) and a["duration_s"] == 2.0
    b = m.step()
    assert b["done"] and b["duration_s"] == 1.5 and m.time_s == 3.5
    assert np.allclose(m.position_m, [3.0, 4.0]) and np.allclose(
        m.lookahead_position_m(), [3.0, 4.0]
    )
    try:
        m.step()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Step after completion accepted.")
    m.reset()
    assert np.allclose(m.position_m, [0.0, 0.0]) and not m.done
    for speed, dt in [(0.0, 0.5), (-1.0, 0.5), (float("nan"), 0.5), (1.0, 0.0)]:
        try:
            RouteMotion([[0.0, 0.0], [1.0, 0.0]], [0.0, 1.0], speed, dt)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid motion setting accepted.")
    try:
        RouteMotion([[0.0, 0.0], [1.0, 0.0]], [0.0, 2.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid route distance accepted.")
    print(
        "Motion tests passed: turns, lookahead, endpoint, final partial step, reset and invalid inputs."
    )
