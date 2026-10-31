"""Envs / features.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import json
import numpy as np
from toponet.data.input_inspection import load_environment_inputs
from toponet.baselines.controllers import A3StyleController

FEATURE_NAMES = (
    "rsrp_normalized",
    "sinr_normalized",
    "candidate_user_rate_fraction",
    "requested_background_pressure_bounded",
    "background_reference_utilization",
    "active_background_sessions_bounded",
    "prb_capacity_normalized",
    "is_serving_cell",
    "is_previous_cell",
    "recent_handover_age_normalized",
    "distance_to_user_bounded",
    "lookahead_distance_bounded",
    "interruption_remaining_bounded",
    "last_interval_cell_utilization",
    "last_interval_user_rate_on_serving_cell",
)

FEATURE_DESCRIPTIONS = (
    "Clip (RSRP+140)/100 to [0,1]; clips signals outside [-140,-40] dBm.",
    "Clip (SINR+20)/60 to [0,1]; clips outside [-20,40] dB.",
    "CURRENT modeled candidate rate / configured user request; 0 if request=0.",
    "p/(1+p), p=current requested background Mbps / nominal full-resource background Mbps; not actual utilization.",
    "Current background-only reference utilization from radio model; not measured operator PRB data.",
    "Current active session count / (count+30); no future arrivals or departure times.",
    "PRBs/100; expected uploaded LTE capacities 50,75,100.",
    "One-hot current serving cell.",
    "One-hot cell left at most recent handover; all zero before first handover.",
    "Clip time since last handover / 3 seconds to [0,1], broadcast; 1 before first handover.",
    "Current horizontal cell-to-user distance / (distance+1000 metres).",
    "Cell distance to the known-route one-step-ahead position / (distance+1000 metres); route lookahead assumed available.",
    "Remaining modeled handover interruption / (remaining+movement dt), broadcast.",
    "Realized modeled resource utilization averaged over last completed step; initialized to current background reference utilization.",
    "Last completed step user Mbps / configured user request on CURRENT serving node, other nodes 0; initial value 0.",
)


@dataclass(frozen=True)
class RewardSettings:
    throughput_scale_mbps: float = 20.0
    throughput_weight: float = 1.0
    handover_cost: float = 0.1
    ping_pong_cost: float = 2.0
    weak_signal_cost_per_s: float = 1.0


def reward_components(info, settings=None):
    f = RewardSettings() if settings is None else settings
    v = list(asdict(f).values())
    if (
        not all(math.isfinite(x) for x in v)
        or f.throughput_scale_mbps <= 0
        or any(x < 0 for x in v[1:])
    ):
        raise ValueError("Invalid reward settings.")
    dt = float(info["duration_s"])
    rate = float(info["user_delivered_mbps"])
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(rate) or rate < 0:
        raise ValueError("Invalid realized throughput or duration.")
    # Integrate ongoing rates over actual time; event costs occur ONCE per event.
    parts = {
        "throughput": f.throughput_weight * rate / f.throughput_scale_mbps * dt,
        "handover": -f.handover_cost * int(bool(info["handover"])),
        "ping_pong": -f.ping_pong_cost * int(bool(info["ping_pong"])),
        "weak_signal": -f.weak_signal_cost_per_s * dt * int(bool(info["weak_signal"])),
    }
    parts["total"] = float(sum(parts.values()))
    return parts


def encode_features(
    obs,
    cell_xy_m,
    prbs,
    bg_capacity_mbps,
    user_request_mbps,
    dt_s,
    last_utilization,
    last_user_mbps,
):
    xy = np.asarray(cell_xy_m, dtype=float)
    prbs = np.asarray(prbs, dtype=float)
    caps = np.asarray(bg_capacity_mbps, dtype=float)
    util = np.asarray(last_utilization, dtype=float)
    n = len(prbs)
    serving = obs["serving_cell"]
    previous = obs["previous_cell"]
    if (
        xy.shape != (n, 2)
        or caps.shape != (n,)
        or util.shape != (n,)
        or not all(np.isfinite(a).all() for a in [xy, prbs, caps, util])
        or (caps <= 0).any()
        or (prbs <= 0).any()
        or (prbs > 100).any()
    ):
        raise ValueError("Invalid feature metadata.")
    if (
        not math.isfinite(user_request_mbps)
        or user_request_mbps < 0
        or not math.isfinite(dt_s)
        or dt_s <= 0
        or not math.isfinite(last_user_mbps)
        or last_user_mbps < 0
    ):
        raise ValueError("Invalid feature scale/history.")
    rsrp = np.asarray(obs["rsrp_dbm"], dtype=float)
    sinr = np.asarray(obs["sinr_db"], dtype=float)
    rates = np.asarray(obs["candidate_user_rate_mbps"], dtype=float)
    demand = np.asarray(obs["background_demand_mbps"], dtype=float)
    counts = np.asarray(obs["active_background_sessions"], dtype=float)
    bgutil = np.asarray(obs["background_reference_utilization"], dtype=float)
    if (
        any(
            a.shape != (n,) or not np.isfinite(a).all()
            for a in [rsrp, sinr, rates, demand, counts, bgutil]
        )
        or (demand < 0).any()
        or (counts < 0).any()
    ):
        raise ValueError("Invalid current observation arrays.")
    if not 0 <= serving < n or (previous is not None and not 0 <= previous < n):
        raise ValueError("Invalid history nodes.")
    current = np.asarray(obs["position_m"], dtype=float)
    ahead = np.asarray(obs["lookahead_position_m"], dtype=float)
    if (
        current.shape != (2,)
        or ahead.shape != (2,)
        or not np.isfinite(current).all()
        or not np.isfinite(ahead).all()
    ):
        raise ValueError("Invalid route positions.")
    age = obs["time_since_handover_s"]
    pause = float(obs["interruption_remaining_s"])
    if (
        (age is not None and (not math.isfinite(age) or age < 0))
        or not math.isfinite(pause)
        or pause < 0
    ):
        raise ValueError("Invalid elapsed times.")
    x = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = np.clip((rsrp + 140.0) / 100.0, 0.0, 1.0)
    x[:, 1] = np.clip((sinr + 20.0) / 60.0, 0.0, 1.0)
    x[:, 2] = (
        np.clip(rates / user_request_mbps, 0.0, 1.0) if user_request_mbps > 0 else 0.0
    )
    pressure = demand / caps
    x[:, 3] = pressure / (1.0 + pressure)
    x[:, 4] = np.clip(bgutil, 0.0, 1.0)
    x[:, 5] = counts / (counts + 30.0)
    x[:, 6] = prbs / 100.0
    x[serving, 7] = 1.0
    if previous is not None:
        x[previous, 8] = 1.0
    x[:, 9] = 1.0 if age is None else np.clip(age / 3.0, 0.0, 1.0)
    dist = np.linalg.norm(xy - current, axis=1)
    next_dist = np.linalg.norm(xy - ahead, axis=1)
    x[:, 10] = dist / (dist + 1000.0)
    x[:, 11] = next_dist / (next_dist + 1000.0)
    x[:, 12] = pause / (pause + dt_s)
    x[:, 13] = np.clip(util, 0.0, 1.0)
    x[serving, 14] = (
        np.clip(last_user_mbps / user_request_mbps, 0.0, 1.0)
        if user_request_mbps > 0
        else 0.0
    )
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise RuntimeError("Features are not finite and bounded.")
    return x


class LearningInterface:
    """Returns state,reward,done,info but contains NO network, replay or learning.

    Rewards are a transparent draft scalar objective; background service stays
    an independently reported metric, not a fictitious network fairness bonus.
    Latent traffic profiles mean this observation is not claimed fully Markov.
    """

    def __init__(self, environment, reward_settings=None):
        self.env = environment
        self.reward_settings = (
            RewardSettings() if reward_settings is None else reward_settings
        )
        self.feature_names = FEATURE_NAMES
        self.graph = self.env.neighbors.astype(np.float32).copy()
        np.fill_diagonal(self.graph, 1.0)
        self.last_utilization = self.env.observe()[
            "background_reference_utilization"
        ].copy()
        self.last_user_mbps = 0.0
        self.reward_total = 0.0
        self.reward_totals = {
            k: 0.0 for k in ("throughput", "handover", "ping_pong", "weak_signal")
        }
        reward_components(
            {
                "duration_s": 1.0,
                "user_delivered_mbps": 0.0,
                "handover": False,
                "ping_pong": False,
                "weak_signal": False,
            },
            self.reward_settings,
        )

    def reset(self, traffic_seed=7, radio_seed=107, spatial_mode="nominal"):
        raw = self.env.reset(traffic_seed, radio_seed, spatial_mode)
        self.last_utilization = raw["background_reference_utilization"].copy()
        self.last_user_mbps = 0.0
        self.reward_total = 0.0
        self.reward_totals = {k: 0.0 for k in self.reward_totals}
        return self.state(raw)

    def state(self, raw=None):
        raw = self.env.observe() if raw is None else raw
        features = encode_features(
            raw,
            self.env.xy_m,
            self.env.radio.prbs,
            self.env.radio.background_capacity_mbps,
            self.env.radio.settings.user_demand_mbps,
            self.env.motion.dt_s,
            self.last_utilization,
            self.last_user_mbps,
        )
        # Only these three arrays go to the learner. No IDs, future events, seed or
        # episode labels are node features. Copies protect simulator/history state.
        return {
            "node_features": features.copy(),
            "action_mask": raw["action_mask"].copy(),
            "message_graph": self.graph.copy(),
        }

    def step(self, action):
        before = self.env.utilization_integral.copy()
        raw, done, info = self.env.step(action)
        self.last_utilization = (self.env.utilization_integral - before) / info[
            "duration_s"
        ]
        self.last_user_mbps = info["user_delivered_mbps"]
        parts = reward_components(info, self.reward_settings)
        self.reward_total += parts["total"]
        for k in self.reward_totals:
            self.reward_totals[k] += parts[k]
        info = {**info, "reward_components": parts}
        return self.state(raw), parts["total"], done, info

    def metadata(self):
        return {
            "feature_count": len(FEATURE_NAMES),
            "features": [
                {"index": i, "name": name, "definition": FEATURE_DESCRIPTIONS[i]}
                for i, name in enumerate(FEATURE_NAMES)
            ],
            "reward_settings": asdict(self.reward_settings),
            "reward_formula": "throughput_weight * (delivered_user_Mbps / throughput_scale_Mbps) * actual_duration_s - handover_cost * HO_event - ping_pong_cost * PP_event - weak_signal_cost_per_s * weak_duration_s",
            "reward_status": "Starting, configurable research choice, NOT optimized or guaranteed to produce zero ping-pongs. Freeze after development; assess sensitivity before final claims.",
            "background_objective": "Background delivered/unmet data reported separately. No global Jain-index bonus: one moving user cannot meaningfully control all 46 background populations.",
            "information_assumptions": "Current simulator-wide radio/load estimates plus known-road-route one-step lookahead. Requires telemetry and route availability; not established for an actual phone.",
            "limitations": "Partial observation: future profiles/arrivals/departures and seed omitted. No claim of Markov sufficiency, real-network calibration, learned traffic prediction, or GNN necessity.",
        }


def self_test():
    base = {
        "duration_s": 0.5,
        "user_delivered_mbps": 20.0,
        "handover": False,
        "ping_pong": False,
        "weak_signal": False,
    }
    assert reward_components(base)["total"] == 0.5
    assert np.isclose(reward_components({**base, "handover": True})["total"], 0.4)
    assert np.isclose(
        reward_components({**base, "handover": True, "ping_pong": True})["total"], -1.6
    )
    assert np.isclose(
        reward_components({**base, "user_delivered_mbps": 0.0, "weak_signal": True})[
            "total"
        ],
        -0.5,
    )
    assert np.isclose(reward_components({**base, "duration_s": 0.2})["total"], 0.2)
    # Ongoing terms integrate identically when splitting time; event penalties do not repeat.
    whole = reward_components({**base, "duration_s": 1.0, "handover": True})["total"]
    split = (
        reward_components({**base, "handover": True})["total"]
        + reward_components(base)["total"]
    )
    assert np.isclose(whole, split)
    from toponet.envs.handover import HandoverEnvironment
    from toponet.data.graph import build_neighbor_graph
    from toponet.data.movement import RouteMotion
    from toponet.envs.radio import RadioSettings

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
    e = HandoverEnvironment(
        cells,
        xy,
        build_neighbor_graph(xy, 200.0)[0],
        RouteMotion([[0.0, 0.0], [100.0, 0.0]], [0.0, 100.0]),
        radio_settings=RadioSettings(shadowing_std_db=0.0),
    )
    w = LearningInterface(e)
    s = w.reset()
    assert (
        s["node_features"].shape == (3, 15)
        and s["action_mask"].shape == (3,)
        and s["message_graph"].shape == (3, 3)
    )
    assert np.diag(s["message_graph"]).all() and s["node_features"][:, 8].sum() == 0.0
    s["action_mask"][:] = False
    s["node_features"][:] = 0.0
    assert w.state()["action_mask"][0]
    ns, r, done, info = w.step(1)
    assert ns["node_features"][1, 7] == 1.0 and ns["node_features"][0, 8] == 1.0
    ns, r, done, info = w.step(0)
    assert info["ping_pong"] and info["reward_components"]["ping_pong"] == -2.0
    assert np.isclose(w.reward_total, sum(w.reward_totals.values()))
    # Joint node permutation preserves features and masks after index remapping.
    raw = e.observe()
    perm = np.array([2, 0, 1])
    inverse = np.argsort(perm)
    permuted = dict(raw)
    for key in (
        "rsrp_dbm",
        "sinr_db",
        "candidate_user_rate_mbps",
        "background_demand_mbps",
        "active_background_sessions",
        "background_reference_utilization",
        "action_mask",
    ):
        permuted[key] = raw[key][perm]
    permuted["serving_cell"] = int(inverse[raw["serving_cell"]])
    permuted["previous_cell"] = int(inverse[raw["previous_cell"]])
    p = encode_features(
        permuted,
        xy[perm],
        e.radio.prbs[perm],
        e.radio.background_capacity_mbps[perm],
        20.0,
        0.5,
        w.last_utilization[perm],
        w.last_user_mbps,
    )
    assert np.allclose(p, w.state()["node_features"][perm])
    # Merely storing a different FUTURE event schedule cannot alter current features.
    saved = e.traffic._events
    e.traffic._events = []
    assert np.array_equal(w.state()["node_features"], ns["node_features"])
    e.traffic._events = saved
    w.reset()
    assert w.reward_total == 0.0 and w.last_user_mbps == 0.0
    print(
        "AI-interface tests passed: feature shapes/bounds, history flags, permutation, copies, future-event exclusion, reward components, time scaling and reset."
    )
