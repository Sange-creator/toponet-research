"""Envs / radio.

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
from toponet.data.movement import RouteMotion
from toponet.envs.traffic import BackgroundTraffic
from toponet.envs.allocation import allocate_cell

SOURCE = "https://www.etsi.org/deliver/etsi_tr/136900_136999/136942/17.00.00_60/tr_136942v170000p.pdf"


@dataclass(frozen=True)
class RadioSettings:
    minimum_horizontal_distance_m: float = 35.0
    assumed_height_above_rooftops_m: float = 15.0
    user_height_m: float = 1.5
    effective_antenna_gain_dbi: float = 0.0
    noise_figure_db: float = 7.0
    subcarrier_spacing_hz: float = 15000.0
    overhead_efficiency: float = 0.75
    spectral_efficiency_cap: float = 6.0
    shadowing_std_db: float = 4.0
    shadowing_correlation_m: float = 20.0
    minimum_activity: float = 0.05
    minimum_target_rsrp_dbm: float = -120.0
    user_demand_mbps: float = 20.0
    background_spectral_efficiency_bps_hz: float = 4.0


class RadioModel:
    def __init__(self, cells, xy_m, seed=107, settings=None):
        self.settings = RadioSettings() if settings is None else settings
        f = self.settings
        values = list(asdict(f).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Radio settings must be finite.")
        if (
            f.minimum_horizontal_distance_m <= 0
            or not 0 < f.assumed_height_above_rooftops_m <= 50
            or f.user_height_m < 0
            or f.subcarrier_spacing_hz <= 0
            or not 0 < f.overhead_efficiency <= 1
            or f.spectral_efficiency_cap <= 0
            or f.shadowing_std_db < 0
            or f.shadowing_correlation_m <= 0
            or not 0 <= f.minimum_activity <= 1
            or f.user_demand_mbps < 0
            or f.background_spectral_efficiency_bps_hz <= 0
        ):
            raise ValueError("Invalid radio settings.")
        self.cells = cells
        self.xy_m = np.asarray(xy_m, dtype=float)
        self.num_cells = len(cells)
        if (
            self.xy_m.shape != (self.num_cells, 2)
            or not np.isfinite(self.xy_m).all()
            or not self.num_cells
        ):
            raise ValueError("Cell coordinates must have shape [cells,2].")
        self.prbs = np.array([c["prb_capacity"] for c in cells], dtype=float)
        self.frequency_mhz = np.array(
            [c["downlink_frequency_mhz"] for c in cells], dtype=float
        )
        self.tx_dbm = np.array([c["tx_power_dbm"] for c in cells], dtype=float)
        self.height_m = np.array([c["height_m"] for c in cells], dtype=float)
        if (
            not all(
                np.isfinite(v).all()
                for v in [self.prbs, self.frequency_mhz, self.tx_dbm, self.height_m]
            )
            or (self.prbs <= 0).any()
            or (self.frequency_mhz <= 0).any()
            or (self.height_m <= 0).any()
        ):
            raise ValueError("Invalid cell radio parameters.")
        self.occupied_hz = self.prbs * 12.0 * f.subcarrier_spacing_hz
        self.per_subcarrier_tx_dbm = self.tx_dbm - 10.0 * np.log10(self.prbs * 12.0)
        self.background_capacity_mbps = (
            self.occupied_hz
            / 1e6
            * f.overhead_efficiency
            * f.background_spectral_efficiency_bps_hz
        )
        self.noise_dbm = (
            -174.0 + 10.0 * np.log10(f.subcarrier_spacing_hz) + f.noise_figure_db
        )
        low = self.frequency_mhz * 1e6 - self.occupied_hz / 2.0
        high = self.frequency_mhz * 1e6 + self.occupied_hz / 2.0
        # Row = candidate serving cell; column = interfering cell. Not generally symmetric.
        overlap_hz = np.maximum(
            0.0,
            np.minimum(high[:, None], high[None, :])
            - np.maximum(low[:, None], low[None, :]),
        )
        self.overlap_fraction = overlap_hz / self.occupied_hz[:, None]
        np.fill_diagonal(self.overlap_fraction, 0.0)
        self.reset(seed)

    def reset(self, seed=107):
        self.rng = np.random.default_rng(seed)
        self.shadowing_db = self.rng.normal(
            0.0, self.settings.shadowing_std_db, self.num_cells
        )

    def advance_shadowing(self, distance_travelled_m):
        d = float(distance_travelled_m)
        if not math.isfinite(d) or d < 0:
            raise ValueError("Travel distance must be finite and nonnegative.")
        if d == 0:
            return
        rho = np.exp(-d / self.settings.shadowing_correlation_m)
        self.shadowing_db = rho * self.shadowing_db + np.sqrt(
            1.0 - rho * rho
        ) * self.rng.normal(0.0, self.settings.shadowing_std_db, self.num_cells)

    def snapshot(self, position_m, background_demand_mbps):
        f = self.settings
        position = np.asarray(position_m, dtype=float)
        demand = np.asarray(background_demand_mbps, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("Invalid moving-user position.")
        if (
            demand.shape != (self.num_cells,)
            or not np.isfinite(demand).all()
            or (demand < 0).any()
        ):
            raise ValueError("Invalid background demand.")
        horizontal = np.maximum(
            np.linalg.norm(self.xy_m - position, axis=1),
            f.minimum_horizontal_distance_m,
        )
        three_d = np.sqrt(horizontal**2 + (self.height_m - f.user_height_m) ** 2)
        h = f.assumed_height_above_rooftops_m
        # TR 36.942 urban NLOS formula; ABOVE-ROOFTOP height is an explicit assumption.
        # Spreadsheet ground-referenced heights cannot be substituted for that quantity.
        empirical = (
            40.0 * (1.0 - 0.004 * h) * np.log10(horizontal / 1000.0)
            - 18.0 * np.log10(h)
            + 21.0 * np.log10(self.frequency_mhz)
            + 80.0
        )
        free_space = (
            32.44
            + 20.0 * np.log10(three_d / 1000.0)
            + 20.0 * np.log10(self.frequency_mhz)
        )
        pathloss = np.maximum(empirical, free_space)
        rsrp = (
            self.per_subcarrier_tx_dbm
            + f.effective_antenna_gain_dbi
            - pathloss
            + self.shadowing_db
        )
        signal_mw = 10.0 ** (rsrp / 10.0)
        background_utilization = np.minimum(1.0, demand / self.background_capacity_mbps)
        activity = (
            f.minimum_activity + (1.0 - f.minimum_activity) * background_utilization
        )
        interference_mw = self.overlap_fraction @ (signal_mw * activity)
        noise_mw = 10.0 ** (self.noise_dbm / 10.0)
        sinr = signal_mw / (interference_mw + noise_mw)
        efficiency = np.minimum(np.log2(1.0 + sinr), f.spectral_efficiency_cap)
        full_rate = self.occupied_hz / 1e6 * f.overhead_efficiency * efficiency
        radio_eligible = rsrp >= f.minimum_target_rsrp_dbm
        candidate_rate = np.zeros(self.num_cells)
        candidate_user_fraction = np.zeros(self.num_cells)
        for i in range(self.num_cells):
            r = allocate_cell(
                float(demand[i]),
                float(self.background_capacity_mbps[i]),
                f.user_demand_mbps,
                float(full_rate[i]) if radio_eligible[i] else 0.0,
            )
            candidate_rate[i] = r["user_delivered_mbps"]
            candidate_user_fraction[i] = r["user_allocated_fraction"]
        if not all(
            np.isfinite(x).all() for x in [rsrp, sinr, full_rate, candidate_rate]
        ):
            raise RuntimeError("Nonfinite radio outputs.")
        return {
            "rsrp_dbm": rsrp,
            "sinr_db": 10.0 * np.log10(np.maximum(sinr, np.finfo(float).tiny)),
            "pathloss_db": pathloss,
            "user_full_resource_rate_mbps": full_rate,
            "candidate_user_rate_mbps": candidate_rate,
            "candidate_user_fraction": candidate_user_fraction,
            "background_utilization": background_utilization,
            "activity": activity,
            "radio_eligible": radio_eligible,
        }


def self_test():
    cells = [
        {
            "prb_capacity": 100,
            "downlink_frequency_mhz": 1815.0,
            "tx_power_dbm": 43.0,
            "height_m": 30.0,
        }
        for _ in range(2)
    ]
    f = RadioSettings(shadowing_std_db=0.0, minimum_activity=0.0)
    m = RadioModel(cells, [[0.0, 0.0], [1000.0, 0.0]], settings=f)
    s = m.snapshot([500.0, 0.0], [54.0, 54.0])
    assert np.allclose(s["rsrp_dbm"][0], s["rsrp_dbm"][1])
    expected_sinr_db = -10.0 * np.log10(
        1.0 + 10.0 ** ((m.noise_dbm - s["rsrp_dbm"][0]) / 10.0)
    )
    assert np.isclose(s["sinr_db"][0], expected_sinr_db) and s["sinr_db"][0] < 0.0
    assert (s["candidate_user_rate_mbps"] <= f.user_demand_mbps).all()
    assert np.allclose(np.diag(m.overlap_fraction), 0.0)
    other = [dict(cells[0]), dict(cells[1])]
    other[1]["downlink_frequency_mhz"] = 1845.0
    nonoverlap = RadioModel(other, [[0.0, 0.0], [1000.0, 0.0]], settings=f)
    assert nonoverlap.overlap_fraction[0, 1] == 0.0
    assert (
        nonoverlap.snapshot([500.0, 0.0], [54.0, 54.0])["sinr_db"][0] > s["sinr_db"][0]
    )
    narrow = [dict(cells[0]), dict(cells[1])]
    narrow[0]["prb_capacity"] = 50
    n = RadioModel(narrow, [[0.0, 0.0], [1000.0, 0.0]], settings=f)
    assert n.overlap_fraction[0, 1] == 1.0 and n.overlap_fraction[1, 0] == 0.5
    assert np.isclose(
        n.per_subcarrier_tx_dbm[0] - n.per_subcarrier_tx_dbm[1], 10.0 * np.log10(2.0)
    )
    far = m.snapshot([5000.0, 0.0], [0.0, 0.0])
    assert far["rsrp_dbm"][0] < s["rsrp_dbm"][0]
    a = RadioModel(cells, [[0.0, 0.0], [1000.0, 0.0]], seed=11)
    b = RadioModel(cells, [[0.0, 0.0], [1000.0, 0.0]], seed=11)
    a.advance_shadowing(5.0)
    b.advance_shadowing(5.0)
    assert np.array_equal(a.shadowing_db, b.shadowing_db)
    before = a.shadowing_db.copy()
    a.advance_shadowing(0.0)
    assert np.array_equal(before, a.shadowing_db)
    assert np.isclose(m.noise_dbm, -125.23908740944319)
    print(
        "Radio tests passed: power/noise units, distance loss, interference, frequency overlap, bandwidth, demand cap and seeded shadowing."
    )
