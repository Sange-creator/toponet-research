"""Envs / allocation.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from pathlib import Path
import csv
import json
import math
import numpy as np


def allocate_cell(
    background_demand_mbps,
    background_capacity_mbps,
    user_demand_mbps=0.0,
    user_capacity_mbps=0.0,
):
    """Proportionally scale requested resource fractions if capacity is exceeded.

    Capacities mean throughput when using ALL resources for that stream.
    Background sessions are aggregated and share an assumed average efficiency.
    A positive request with zero capacity is unserviceable, not free service.
    No interruption is applied here; the later environment handles that interval.
    """
    demand = np.array([background_demand_mbps, user_demand_mbps], dtype=float)
    capacity = np.array([background_capacity_mbps, user_capacity_mbps], dtype=float)
    if (
        not np.isfinite(demand).all()
        or not np.isfinite(capacity).all()
        or (demand < 0).any()
        or (capacity < 0).any()
    ):
        raise ValueError(
            "Demand and full-resource capacities must be finite and nonnegative."
        )
    serviceable = capacity > 0
    requested_resources = np.zeros(2)
    np.divide(demand, capacity, out=requested_resources, where=serviceable)
    if not np.isfinite(requested_resources).all():
        raise ValueError("Resource demand overflows; inspect capacity units.")
    schedulable_pressure = float(requested_resources.sum())
    if not math.isfinite(schedulable_pressure):
        raise ValueError("Total resource demand is not finite.")
    fraction = requested_resources / max(1.0, schedulable_pressure)
    delivered = np.minimum(demand, fraction * capacity)
    unmet = np.maximum(0.0, demand - delivered)
    utilization = float(fraction.sum())
    if utilization > 1.0 + 1e-12 or (delivered > demand + 1e-10).any():
        raise RuntimeError("Invalid resource allocation.")
    return {
        "background_delivered_mbps": float(delivered[0]),
        "user_delivered_mbps": float(delivered[1]),
        "background_unmet_mbps": float(unmet[0]),
        "user_unmet_mbps": float(unmet[1]),
        "background_allocated_fraction": float(fraction[0]),
        "user_allocated_fraction": float(fraction[1]),
        "utilization": min(1.0, utilization),
        "schedulable_resource_pressure": schedulable_pressure,
        "background_unserviceable": bool(demand[0] > 0 and not serviceable[0]),
        "user_unserviceable": bool(demand[1] > 0 and not serviceable[1]),
    }


def self_test():
    a = allocate_cell(20.0, 50.0, 10.0, 50.0)
    assert np.isclose(a["utilization"], 0.6)
    assert a["background_delivered_mbps"] == 20.0 and a["user_delivered_mbps"] == 10.0
    b = allocate_cell(50.0, 50.0, 20.0, 40.0)
    assert np.isclose(b["utilization"], 1.0)
    assert np.isclose(b["background_delivered_mbps"], 100.0 / 3.0)
    assert np.isclose(b["user_delivered_mbps"], 40.0 / 3.0)
    assert np.isclose(b["background_delivered_mbps"] + b["background_unmet_mbps"], 50.0)
    assert np.isclose(b["user_delivered_mbps"] + b["user_unmet_mbps"], 20.0)
    c = allocate_cell(80.0, 50.0)
    assert c["background_delivered_mbps"] == 50.0 and c["background_unmet_mbps"] == 30.0
    d = allocate_cell(10.0, 0.0, 5.0, 50.0)
    assert d["background_unserviceable"] and d["background_delivered_mbps"] == 0.0
    assert d["user_delivered_mbps"] == 5.0
    z = allocate_cell(0.0, 0.0)
    assert z["utilization"] == 0.0
    for demand, capacity in [(-1.0, 50.0), (1.0, -1.0), (float("nan"), 50.0)]:
        try:
            allocate_cell(demand, capacity)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid resource input accepted.")
    rng = np.random.default_rng(7)
    for _ in range(500):
        db, du = rng.uniform(0.0, 150.0, 2)
        cb, cu = rng.uniform(0.1, 100.0, 2)
        r = allocate_cell(db, cb, du, cu)
        assert 0 <= r["utilization"] <= 1.0
        assert np.isclose(
            r["background_delivered_mbps"] / cb + r["user_delivered_mbps"] / cu,
            r["utilization"],
        )
        assert np.isclose(
            r["background_delivered_mbps"] + r["background_unmet_mbps"], db
        )
        assert np.isclose(r["user_delivered_mbps"] + r["user_unmet_mbps"], du)
    print(
        "Resource tests passed: spare capacity, overload, sharing, zero capacity, demand conservation and 500 randomized checks."
    )
