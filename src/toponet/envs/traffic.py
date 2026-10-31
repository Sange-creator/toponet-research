"""Envs / traffic.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import numpy as np
from toponet.data.towers import load_cells


@dataclass(frozen=True)
class TrafficSettings:
    mean_session_duration_s: float = 30.0
    minimum_request_mbps: float = 0.5
    maximum_request_mbps: float = 2.0
    minimum_profile_gap_s: float = 20.0
    maximum_profile_gap_s: float = 60.0
    episode_multiplier_min: float = 0.75
    episode_multiplier_max: float = 1.25
    time_multiplier_min: float = 0.5
    time_multiplier_max: float = 1.5
    reference_overhead_efficiency: float = 0.75
    reference_spectral_efficiency_bps_hz: float = 4.0


class BackgroundTraffic:
    """External traffic generated from a scenario seed, independent of handovers.

    Future events are internal simulator data. Feed only CURRENT/measured state
    to the agent, never future arrivals, profile values, or departure times.
    """

    def __init__(
        self, cells, duration_s, seed=7, settings=None, spatial_mode="nominal"
    ):
        self.settings = TrafficSettings() if settings is None else settings
        f = self.settings
        self.num_cells = len(cells)
        self.duration_s = float(duration_s)
        if (
            not self.num_cells
            or not math.isfinite(self.duration_s)
            or self.duration_s <= 0
        ):
            raise ValueError("Provide cells and a positive finite scenario duration.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        if spatial_mode not in ("nominal", "shuffled"):
            raise ValueError("Spatial mode must be nominal or shuffled.")
        positive = (
            f.mean_session_duration_s,
            f.minimum_request_mbps,
            f.maximum_request_mbps,
            f.minimum_profile_gap_s,
            f.maximum_profile_gap_s,
            f.episode_multiplier_min,
            f.episode_multiplier_max,
            f.time_multiplier_min,
            f.time_multiplier_max,
            f.reference_overhead_efficiency,
            f.reference_spectral_efficiency_bps_hz,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Traffic parameters must be finite and positive.")
        if not (
            f.minimum_request_mbps <= f.maximum_request_mbps
            and f.minimum_profile_gap_s <= f.maximum_profile_gap_s
            and f.episode_multiplier_min <= f.episode_multiplier_max
            and f.time_multiplier_min <= f.time_multiplier_max
            and f.reference_overhead_efficiency <= 1
        ):
            raise ValueError("Inconsistent traffic ranges.")
        priors = np.array([c["assumed_load_fraction"] for c in cells], dtype=float)
        prbs = np.array([c["prb_capacity"] for c in cells], dtype=float)
        if (
            not np.isfinite(priors).all()
            or ((priors < 0) | (priors > 1)).any()
            or not np.isfinite(prbs).all()
            or (prbs <= 0).any()
        ):
            raise ValueError("Invalid assumed priors or resource capacities.")
        # Occupied LTE bandwidth = PRBs * 180 kHz. This is a scaling reference only.
        self.reference_capacity_mbps = (
            prbs
            * 0.180
            * f.reference_overhead_efficiency
            * f.reference_spectral_efficiency_bps_hz
        )
        spatial_seed, profile_seed, session_seed = np.random.SeedSequence(seed).spawn(3)
        spatial_rng = np.random.default_rng(spatial_seed)
        profile_rng = np.random.default_rng(profile_seed)
        self.spatial_mode = spatial_mode
        self.assumed_priors = (
            priors.copy()
            if spatial_mode == "nominal"
            else spatial_rng.permutation(priors)
        )
        episode_scale = spatial_rng.uniform(
            f.episode_multiplier_min, f.episode_multiplier_max, size=self.num_cells
        )
        mean_request = 0.5 * (f.minimum_request_mbps + f.maximum_request_mbps)
        self.base_arrival_rates = (
            self.assumed_priors
            * self.reference_capacity_mbps
            * episode_scale
            / (mean_request * f.mean_session_duration_s)
        )
        # A common time grid, but independent random multiplier values for each cell.
        times = [0.0]
        while times[-1] < self.duration_s:
            times.append(
                min(
                    self.duration_s,
                    times[-1]
                    + profile_rng.uniform(
                        f.minimum_profile_gap_s, f.maximum_profile_gap_s
                    ),
                )
            )
        self.profile_times_s = np.array(times)
        multipliers = profile_rng.uniform(
            f.time_multiplier_min,
            f.time_multiplier_max,
            size=(len(times), self.num_cells),
        )
        multipliers[0] = 1.0
        self.profile_rates = multipliers * self.base_arrival_rates
        rng = np.random.default_rng(session_seed)
        initial_counts = rng.poisson(self.profile_rates[0] * f.mean_session_duration_s)
        initial_demands = np.zeros(self.num_cells)
        events = []
        for cell in range(self.num_cells):
            requests = rng.uniform(
                f.minimum_request_mbps,
                f.maximum_request_mbps,
                int(initial_counts[cell]),
            )
            residual_durations = np.maximum(
                rng.exponential(f.mean_session_duration_s, len(requests)), 1e-9
            )
            initial_demands[cell] = requests.sum()
            for end, request in zip(residual_durations, requests):
                if end <= self.duration_s:
                    events.append((float(end), -1, cell, float(request)))
            # Exact thinning for each piecewise-linear arrival-rate interval.
            for j, (start, end) in enumerate(
                zip(self.profile_times_s[:-1], self.profile_times_s[1:])
            ):
                low, high = self.profile_rates[j, cell], self.profile_rates[j + 1, cell]
                bound = max(low, high)
                if bound <= 0:
                    continue
                candidate_count = int(rng.poisson(bound * (end - start)))
                candidates = rng.uniform(start, end, candidate_count)
                rates = low + (high - low) * (candidates - start) / (end - start)
                accepted = candidates[rng.random(candidate_count) < rates / bound]
                requests = rng.uniform(
                    f.minimum_request_mbps, f.maximum_request_mbps, len(accepted)
                )
                durations = np.maximum(
                    rng.exponential(f.mean_session_duration_s, len(accepted)), 1e-9
                )
                for arrival, request, lifetime in zip(accepted, requests, durations):
                    arrival = float(arrival)
                    request = float(request)
                    events.append((arrival, 1, cell, request))
                    departure = arrival + float(lifetime)
                    if departure <= self.duration_s:
                        events.append((departure, -1, cell, request))
        self._events = sorted(events)
        self._initial_counts = initial_counts.astype(np.int64)
        self._initial_demands = initial_demands
        self.reset()

    def reset(self):
        self.time_s = 0.0
        self.active_counts = self._initial_counts.copy()
        self.current_demand_mbps = self._initial_demands.copy()
        self.last_mean_demand_mbps = self.current_demand_mbps.copy()
        self.last_mean_active_counts = self.active_counts.astype(float)
        self._event_index = 0
        self.total_arrivals = 0
        self.total_departures = 0

    def advance(self, duration_s):
        duration = float(duration_s)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Advance duration must be positive and finite.")
        end = self.time_s + duration
        if end > self.duration_s + 1e-8:
            raise ValueError("Traffic scenario does not cover this interval.")
        end = min(end, self.duration_s)
        actual_duration = end - self.time_s
        if actual_duration <= 0:
            raise RuntimeError("Traffic scenario has finished.")
        cursor = self.time_s
        demand_integral = np.zeros(self.num_cells)
        count_integral = np.zeros(self.num_cells)
        while (
            self._event_index < len(self._events)
            and self._events[self._event_index][0] <= end
        ):
            time, change, cell, request = self._events[self._event_index]
            gap = time - cursor
            if gap < -1e-9:
                raise RuntimeError("Events are not in chronological order.")
            demand_integral += self.current_demand_mbps * max(0.0, gap)
            count_integral += self.active_counts * max(0.0, gap)
            self.active_counts[cell] += change
            self.current_demand_mbps[cell] += change * request
            if self.active_counts[cell] < 0 or self.current_demand_mbps[cell] < -1e-8:
                raise RuntimeError("Invalid session accounting.")
            if self.active_counts[cell] == 0:
                self.current_demand_mbps[cell] = 0.0
            if change > 0:
                self.total_arrivals += 1
            else:
                self.total_departures += 1
            cursor = time
            self._event_index += 1
        demand_integral += self.current_demand_mbps * (end - cursor)
        count_integral += self.active_counts * (end - cursor)
        self.last_mean_demand_mbps = demand_integral / actual_duration
        self.last_mean_active_counts = count_integral / actual_duration
        self.time_s = end
        return {
            "duration_s": actual_duration,
            "mean_demand_mbps": self.last_mean_demand_mbps.copy(),
            "mean_active_counts": self.last_mean_active_counts.copy(),
        }

    def current_arrival_rates(self):
        t = min(self.time_s, self.duration_s)
        j = min(
            int(np.searchsorted(self.profile_times_s, t, side="right")) - 1,
            len(self.profile_times_s) - 2,
        )
        fraction = (t - self.profile_times_s[j]) / (
            self.profile_times_s[j + 1] - self.profile_times_s[j]
        )
        return (
            self.profile_rates[j] * (1 - fraction)
            + self.profile_rates[j + 1] * fraction
        )


def self_test():
    cell = {"assumed_load_fraction": 0.5, "prb_capacity": 100}
    a = BackgroundTraffic([cell], 10.0, seed=7)
    a.active_counts[:] = 1
    a.current_demand_mbps[:] = 1.0
    a._events = [(0.25, -1, 0, 1.0), (0.75, 1, 0, 2.0)]
    a._event_index = 0
    result = a.advance(1.0)
    assert np.allclose(result["mean_demand_mbps"], [0.75])
    assert np.allclose(result["mean_active_counts"], [0.5])
    assert a.active_counts[0] == 1 and a.current_demand_mbps[0] == 2.0
    b = BackgroundTraffic([cell], 10.0, seed=17)
    c = BackgroundTraffic([cell], 10.0, seed=17)
    assert b._events == c._events and np.array_equal(b.active_counts, c.active_counts)
    b_result = b.advance(2.0)
    c1 = c.advance(0.5)
    c2 = c.advance(1.5)
    assert np.array_equal(b.active_counts, c.active_counts)
    assert np.allclose(b.current_demand_mbps, c.current_demand_mbps)
    assert np.allclose(
        b_result["mean_demand_mbps"] * 2.0,
        c1["mean_demand_mbps"] * 0.5 + c2["mean_demand_mbps"] * 1.5,
    )
    b.reset()
    assert b.time_s == 0 and b._event_index == 0
    empty = BackgroundTraffic(
        [{"assumed_load_fraction": 0.0, "prb_capacity": 50}], 10.0
    )
    empty.advance(10.0)
    assert empty.active_counts[0] == 0 and empty.current_demand_mbps[0] == 0.0
    print(
        "Traffic tests passed: exact interval averages, session accounting, seed reproducibility, interval splitting, reset and zero demand."
    )
