"""Evaluation / diagnostics.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

import copy, csv, gzip, hashlib, json, math
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def csvout(p, rows):
    if not rows:
        return
    with (
        gzip.open(p, "wt", newline="")
        if str(p).endswith(".gz")
        else open(p, "w", newline="") as f
    ):
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def external(env, raw=None):
    raw = env.observe() if raw is None else raw
    return [
        raw["position_m"],
        env.radio.shadowing_db,
        raw["background_demand_mbps"],
        raw["active_background_sessions"],
    ]


def digest_add(h, env, raw):
    for v in external(env, raw):
        h.update(np.asarray(v, dtype="<f8").tobytes())


def interventions(env, target, windows=(0.5, 1.0, 5.0, 10.0)):
    """Identical pre-action state; each branch holds its initial assigned cell.
    Not the realized policy continuation, not a comparison to optimal stay/switch.
    """
    old = env.serving_cell
    if old == target:
        return []
    before = external(env)
    a = copy.deepcopy(env)
    b = copy.deepcopy(env)
    available = (
        env.motion.route_length_m - env.motion.distance_m
    ) / env.motion.speed_mps
    actual = sorted(set(min(float(w), available) for w in windows))
    results = {}
    start = env.motion.time_s
    while not a.motion.done and a.motion.time_s - start < max(actual) - 1e-8:
        for x, y in zip(external(a), external(b)):
            if not np.array_equal(x, y):
                raise RuntimeError("Counterfactual external state diverged")
        # Native dt retained; partial endpoint only. Requested windows align to dt=.5.
        a.step(int(target))
        b.step(int(old))
        t = a.motion.time_s - start
        for horizon in actual:
            if horizon not in results and t >= horizon - 1e-8:
                if not math.isclose(t, horizon, abs_tol=1e-7):
                    raise ValueError("Counterfactual horizon not aligned to native dt")
                results[horizon] = dict(
                    actual_window_s=t,
                    switch_mbit=a.user_delivered_mbit - env.user_delivered_mbit,
                    stay_mbit=b.user_delivered_mbit - env.user_delivered_mbit,
                    delta_mbit=a.user_delivered_mbit - b.user_delivered_mbit,
                    switch_weak_s=a.weak_duration_s - env.weak_duration_s,
                    stay_weak_s=b.weak_duration_s - env.weak_duration_s,
                    switch_handover_count=a.total_handovers - env.total_handovers,
                    stay_handover_count=b.total_handovers - env.total_handovers,
                )
    for x, y in zip(before, external(env)):
        if not np.array_equal(x, y):
            raise RuntimeError("Interventions mutated original environment")
    return [
        dict(requested_window_s=w, **results[min(float(w), available)]) for w in windows
    ]


def stays(rows):
    if not rows:
        return []
    starts = [0] + [i for i, x in enumerate(rows) if x["handover"] and i > 0]
    out = []
    for j, start in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(rows)
        part = rows[start:end]
        first = part[0]
        dt = sum(x["duration_s"] for x in part)
        entered = bool(first["handover"])
        completed = entered and end < len(rows)
        prev = first["left_site_id"] if entered else None
        item = dict(
            segment=j,
            site_id=first["serving_site_id"],
            previous_site_id=prev,
            entered_by_handover=entered,
            completed=completed,
            right_censored=end == len(rows),
            dwell_s=dt,
            strict_under5=completed and dt < 5 - 1e-8,
            entry_rsrp_dbm=first["serving_rsrp_dbm"],
            exit_sample_rsrp_dbm=part[-1]["serving_rsrp_dbm"],
            delivered_mbit=sum(x["user_mbps"] * x["duration_s"] for x in part),
            weak_s=sum(x["duration_s"] * x["weak"] for x in part),
        )
        for name, key, threshold in [
            ("target_gt_previous_s", "rsrp_advantage_previous_db", 0.0),
            ("target_gt_previous_plus3_s", "rsrp_advantage_previous_db", 3.0),
        ]:
            item[name] = sum(
                x["duration_s"] for x in part if entered and x[key] > threshold
            )
        for threshold in [-120, -110, -100]:
            item[f"rsrp_ge_{abs(threshold)}_s"] = sum(
                x["duration_s"] for x in part if x["serving_rsrp_dbm"] >= threshold
            )
        item["relative_stronger_fraction"] = (
            item["target_gt_previous_s"] / dt if entered else None
        )
        if end < len(rows):
            nxt = rows[end]
            item.update(
                next_target_site_id=nxt["serving_site_id"],
                next_rsrp_gain_db=nxt["entry_rsrp_gain_db"],
                next_candidate_rate_gain_mbps=nxt["entry_candidate_rate_gain_mbps"],
                departing_rsrp_dbm=nxt["left_rsrp_dbm"],
            )
        else:
            item.update(
                next_target_site_id=None,
                next_rsrp_gain_db=None,
                next_candidate_rate_gain_mbps=None,
                departing_rsrp_dbm=None,
            )
        out.append(item)
    return out


def rollout(vi, agent, traffic, radio, route, controller=None, counterfactual=True):
    import torch
    from toponet.training.episode import learning_snapshot, tree_equal

    before = learning_snapshot(agent) if agent is not None else None
    state = vi.reset(traffic, radio, "nominal")
    env = vi.env
    if controller is not None:
        controller.reset()
    rows = []
    cf = []
    h = hashlib.sha256()
    previous = None
    arrays = {
        k: []
        for k in [
            "rsrp_dbm",
            "sinr_db",
            "candidate_user_rate_mbps",
            "background_demand_mbps",
            "background_reference_utilization",
            "realized_modeled_utilization",
        ]
    }
    while True:
        raw = env.observe()
        digest_add(h, env, raw)
        old = int(raw["serving_cell"])
        action = int(
            agent.select_action(state, eval_mode=True)
            if agent is not None
            else controller.act(raw)
        )
        if not raw["action_mask"][action]:
            raise RuntimeError("Illegal frozen action")
        if action != old:
            previous = old
            if counterfactual:
                for v in interventions(env, action):
                    cf.append(
                        dict(
                            route=route,
                            event_time_s=float(raw["time_s"]),
                            from_site_id=env.cells[old]["site_id"],
                            to_site_id=env.cells[action]["site_id"],
                            **v,
                        )
                    )
        prevutil = env.utilization_integral.copy()
        state, reward, done, info = vi.step(action)
        dt = float(info["duration_s"])
        util = (env.utilization_integral - prevutil) / dt
        if np.any(util < -1e-8) or np.any(util > 1 + 1e-8):
            raise RuntimeError("Utilization outside range")
        if bool(info["handover"]) != (old != action):
            raise RuntimeError("Action execution mismatch")
        row = dict(
            route=route,
            time_s=float(info["decision_time_s"]),
            end_time_s=float(info["end_time_s"]),
            duration_s=dt,
            serving_site_id=env.cells[action]["site_id"],
            left_site_id=env.cells[old]["site_id"],
            previous_site_id=env.cells[previous]["site_id"]
            if previous is not None
            else None,
            handover=bool(info["handover"]),
            ping_pong=bool(info["ping_pong"]),
            serving_rsrp_dbm=float(raw["rsrp_dbm"][action]),
            serving_sinr_db=float(raw["sinr_db"][action]),
            left_rsrp_dbm=float(raw["rsrp_dbm"][old]),
            rsrp_advantage_previous_db=float(
                raw["rsrp_dbm"][action] - raw["rsrp_dbm"][previous]
            )
            if previous is not None
            else 0.0,
            entry_rsrp_gain_db=float(raw["rsrp_dbm"][action] - raw["rsrp_dbm"][old]),
            entry_candidate_rate_gain_mbps=float(
                raw["candidate_user_rate_mbps"][action]
                - raw["candidate_user_rate_mbps"][old]
            ),
            user_mbps=float(info["user_delivered_mbps"]),
            weak=bool(info["weak_signal"]),
            serving_modeled_utilization=float(util[action]),
            network_mean_modeled_utilization=float(util.mean()),
            background_delivered_mbps=float(info["background_delivered_mbps"]),
            interruption_before_s=float(raw["interruption_remaining_s"]),
        )
        rows.append(row)
        for k in arrays:
            arrays[k].append(
                util.copy() if k == "realized_modeled_utilization" else raw[k].copy()
            )
        if done:
            break
    if before is not None and not tree_equal(before, learning_snapshot(agent)):
        raise RuntimeError("Frozen learner changed")
    segments = stays(rows)
    durations = [x["dwell_s"] for x in segments if x["completed"]]
    result = dict(
        env.summary(),
        route=route,
        traffic_seed=traffic,
        radio_seed=radio,
        steps=len(rows),
        trajectory_complete=True,
        short_stays=sum(x < 5 - 1e-8 for x in durations),
        completed_stays=len(durations),
        external_state_sha256=h.hexdigest(),
    )
    for name in ["rsrp_dbm", "sinr_db"]:
        key = "serving_" + name
        result["duration_weighted_" + name] = (
            sum(x[key] * x["duration_s"] for x in rows) / result["elapsed_s"]
        )
    result["duration_weighted_serving_modeled_utilization"] = (
        sum(x["serving_modeled_utilization"] * x["duration_s"] for x in rows)
        / result["elapsed_s"]
    )
    result["completed_stays_B_gt_A_at_least80pct"] = sum(
        x["relative_stronger_fraction"] >= 0.8 for x in segments if x["completed"]
    )
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    arrays["site_ids"] = np.array([c["site_id"] for c in env.cells])
    arrays["time_s"] = np.array([x["time_s"] for x in rows])
    arrays["duration_s"] = np.array([x["duration_s"] for x in rows])
    return result, rows, segments, cf, arrays


def match(actual, expected):
    for key in [
        "route",
        "traffic_seed",
        "radio_seed",
        "steps",
        "initial_serving_site_id",
        "external_state_sha256",
        "handovers",
        "ping_pongs",
        "short_stays",
        "completed_stays",
    ]:
        if actual[key] != expected[key]:
            raise RuntimeError("Archived mismatch: " + key)
    for key in [
        "elapsed_s",
        "user_delivered_mbit",
        "weak_signal_duration_s",
        "mean_user_delivered_mbps",
        "mean_cell_utilization",
        "background_delivered_mbit",
        "background_unmet_mbit",
    ]:
        if not np.allclose(actual[key], expected[key], atol=1e-6, rtol=1e-9):
            raise RuntimeError("Archived mismatch: " + key)
