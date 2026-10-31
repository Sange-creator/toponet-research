"""Reproducible paired descriptive analysis of the verified historical cases."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from toponet.evaluation.metrics import aggregate
from toponet.io import sha256, write_json


def cluster_id(area, scenario):
    if area == "pokhara":
        return scenario["route"]  # Keep repeats across layouts/conditions together.
    match = re.search(r"_from(\d+)_to(\d+)", scenario["route"])
    if not match:
        raise ValueError("Cannot determine OD group: " + scenario["route"])
    return "-".join(map(str, sorted(map(int, match.groups()))))


def cluster_bootstrap(cases, scenarios, label, area, samples):
    groups = defaultdict(list)
    for case, scenario in zip(cases, scenarios):
        groups[cluster_id(area, scenario)].append(case)

    # Exposure, user delivery, handovers, ping-pongs, weak time, short stays, stays.
    def totals(rows, name):
        return [
            sum(c[name][k] for c in rows)
            for k in (
                "elapsed_s",
                "user_delivered_mbit",
                "handovers",
                "ping_pongs",
                "weak_signal_duration_s",
                "short_stays",
                "completed_stays",
            )
        ]

    model = np.array([totals(rows, label) for rows in groups.values()], dtype=float)
    base = np.array([totals(rows, "a3") for rows in groups.values()], dtype=float)
    rng = np.random.default_rng(20261029)
    draws = rng.integers(0, len(groups), size=(samples, len(groups)))
    m = model[draws].sum(axis=1)
    b = base[draws].sum(axis=1)
    values = dict(
        throughput_delta_mbps=m[:, 1] / m[:, 0] - b[:, 1] / b[:, 0],
        handover_rate_delta_per_hour=3600 * (m[:, 2] / m[:, 0] - b[:, 2] / b[:, 0]),
        ping_pong_rate_delta_per_hour=3600 * (m[:, 3] / m[:, 0] - b[:, 3] / b[:, 0]),
        weak_time_delta_percentage_points=100 * (m[:, 4] / m[:, 0] - b[:, 4] / b[:, 0]),
        short_stay_delta_percentage_points=100
        * (m[:, 5] / m[:, 6] - b[:, 5] / b[:, 6]),
    )
    return dict(
        groups=len(groups),
        samples=samples,
        seed=20261029,
        interval_95={
            key: np.quantile(v, [0.025, 0.975]).tolist() for key, v in values.items()
        },
        limitation="Conditional on this frozen policy and these layouts/conditions. OD/route clustering does not remove all correlation from shared roads; not a confidence interval over training seeds, cities or layout populations.",
    )


def analyze(project, samples=2000):
    if samples < 100:
        raise ValueError("At least 100 bootstrap draws required")
    project = Path(project)
    summaries = []
    paired = []
    output = {
        "scope": "Historical evaluation; no policy/seed selection or training",
        "areas": {},
    }
    for area in ("heldout", "pokhara"):
        folder = project / "evidence" / area
        plan = json.loads(
            (project / "evidence/heldout/EVALUATION_PLAN.json").read_text()
        )
        scenarios = (
            plan["scenarios"]
            if area == "heldout"
            else json.loads((project / "data/pokhara/SCENARIOS.json").read_text())
        )
        cases = []
        for i, scenario in enumerate(scenarios, 1):
            case_dir = folder / "cases" / f"case_{i:04d}"
            marker = json.loads((case_dir / "COMPLETE.json").read_text())
            if sha256(case_dir / "results.json") != marker["files"]["results.json"]:
                raise ValueError("Historical result hash mismatch")
            if area == "heldout" and marker["route"] != scenario["route"]:
                raise ValueError("Historical route mismatch")
            if area == "pokhara" and marker["scenario"] != scenario:
                raise ValueError("Historical scenario mismatch")
            values = json.loads((case_dir / "results.json").read_text())
            if set(values) != set(plan["controllers"]):
                raise ValueError("Unexpected controllers")
            for label, row in values.items():
                for key in ("route", "traffic_seed", "radio_seed"):
                    if row[key] != scenario[key]:
                        raise ValueError("Result scenario mismatch")
                for key in (
                    "elapsed_s",
                    "external_state_sha256",
                    "initial_serving_site_id",
                    "steps",
                ):
                    if row[key] != values["a3"][key]:
                        raise ValueError("Pairing mismatch")
                if row.get("trajectory_complete") is not True:
                    raise ValueError("Truncated trajectory")
                if label != "a3":
                    paired.append(
                        dict(
                            area=area,
                            case=i,
                            route=scenario["route"],
                            cluster=cluster_id(area, scenario),
                            controller=label,
                            layout=scenario.get("layout_seed", ""),
                            throughput_delta_mbps=row["mean_user_delivered_mbps"]
                            - values["a3"]["mean_user_delivered_mbps"],
                            handover_delta=row["handovers"] - values["a3"]["handovers"],
                            duration_s=row["elapsed_s"],
                        )
                    )
            cases.append(values)
        metrics = {
            label: aggregate([c[label] for c in cases]) for label in plan["controllers"]
        }
        stats = {}
        for label, m in metrics.items():
            gain = 100 * (m["throughput_mbps"] / metrics["a3"]["throughput_mbps"] - 1)
            reduction = 100 * (1 - m["handovers"] / metrics["a3"]["handovers"])
            summaries.append(
                dict(
                    area=area,
                    controller=label,
                    **m,
                    throughput_gain_percent=gain,
                    handover_reduction_percent=reduction,
                )
            )
            if label != "a3":
                deltas = [
                    c[label]["mean_user_delivered_mbps"]
                    - c["a3"]["mean_user_delivered_mbps"]
                    for c in cases
                ]
                stats[label] = dict(
                    throughput_gain_percent=gain,
                    handover_reduction_percent=reduction,
                    scenario_wins=sum(x > 1e-9 for x in deltas),
                    scenario_losses=sum(x < -1e-9 for x in deltas),
                    scenario_ties=sum(abs(x) <= 1e-9 for x in deltas),
                    clustered_intervals=cluster_bootstrap(
                        cases, scenarios, label, area, samples
                    ),
                )
        layouts = {}
        if area == "pokhara":
            for layout in sorted({s["layout_seed"] for s in scenarios}):
                group = [
                    c for c, s in zip(cases, scenarios) if s["layout_seed"] == layout
                ]
                layouts[str(layout)] = {
                    label: aggregate([c[label] for c in group])
                    for label in plan["controllers"]
                }
        output["areas"][area] = dict(
            cases=len(cases), metrics=metrics, comparisons=stats, by_layout=layouts
        )
    for name, rows in [("summary.csv", summaries), ("paired_scenarios.csv", paired)]:
        with (project / "reports" / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(project / "reports/analysis.json", output)
    plots = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["A3", "Seed 17", "Seed 27", "Seed 37"]
        colors = ["#778899", "#2463a6", "#1e805f", "#ac6332"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for row, area in enumerate(("heldout", "pokhara")):
            metrics = output["areas"][area]["metrics"]
            for column, (key, title) in enumerate(
                [
                    ("throughput_mbps", "Throughput (Mbps)"),
                    ("handovers_per_hour", "Handovers / hour"),
                    ("short_stay_percent", "Short completed stays (%)"),
                ]
            ):
                ax = axes[row, column]
                ax.bar(labels, [m[key] for m in metrics.values()], color=colors)
                ax.set_title(
                    ("Same-area test" if area == "heldout" else "Pokhara transfer")
                    + "\n"
                    + title
                )
                ax.spines[["top", "right"]].set_visible(False)
        for extension in ("png", "pdf"):
            path = project / "reports" / ("performance." + extension)
            fig.savefig(path, dpi=180)
            plots.append(str(path))
        plt.close(fig)
    except ImportError:
        output["plot_note"] = "Install the analysis extra to generate figures."
    return dict(
        areas={k: v["cases"] for k, v in output["areas"].items()},
        analysis=str(project / "reports/analysis.json"),
        figures=plots,
    )
