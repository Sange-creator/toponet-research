#!/usr/bin/env python3
"""Evaluate selected graph-free MLP policies on the frozen paired case sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toponet.agents.ddqn import DQNAgent
from toponet.data.datasets import OriginalDataset, pokhara_interface, read_json
from toponet.envs.features import FEATURE_NAMES
from toponet.evaluation.analysis import cluster_bootstrap, cluster_id
from toponet.evaluation.diagnostics import rollout
from toponet.evaluation.metrics import aggregate
from toponet.evaluation.transfer import TransferPolicy
from toponet.io import sha256, write_json
from nodewise_mlp import NodewiseMLPQNetwork, install_nodewise_mlp


PAIR_KEYS = (
    "route",
    "traffic_seed",
    "radio_seed",
    "steps",
    "elapsed_s",
    "initial_serving_site_id",
    "external_state_sha256",
)


def paired_cluster_intervals(
    cases: list[dict], scenarios: list[dict], model_label: str,
    reference_label: str, area: str, samples: int,
) -> dict:
    """Bootstrap paired route/OD clusters for an MLP-versus-GAT contrast."""
    groups: dict[str, list[dict]] = {}
    for case, scenario in zip(cases, scenarios):
        groups.setdefault(cluster_id(area, scenario), []).append(case)

    def totals(rows: list[dict], label: str) -> list[float]:
        return [
            sum(row[label][key] for row in rows)
            for key in (
                "elapsed_s",
                "user_delivered_mbit",
                "handovers",
                "ping_pongs",
                "weak_signal_duration_s",
                "short_stays",
                "completed_stays",
            )
        ]

    model = np.asarray(
        [totals(rows, model_label) for rows in groups.values()], dtype=float
    )
    reference = np.asarray(
        [totals(rows, reference_label) for rows in groups.values()], dtype=float
    )
    rng = np.random.default_rng(20261030)
    draws = rng.integers(0, len(groups), size=(samples, len(groups)))
    m = model[draws].sum(axis=1)
    r = reference[draws].sum(axis=1)
    if (m[:, 0] <= 0).any() or (r[:, 0] <= 0).any():
        raise ValueError("Bootstrap draw has no exposure")
    if (m[:, 6] <= 0).any() or (r[:, 6] <= 0).any():
        raise ValueError("Bootstrap draw has no completed stays")

    values = {
        "throughput_delta_mbps": m[:, 1] / m[:, 0] - r[:, 1] / r[:, 0],
        "handover_rate_delta_per_hour": 3600
        * (m[:, 2] / m[:, 0] - r[:, 2] / r[:, 0]),
        "ping_pong_rate_delta_per_hour": 3600
        * (m[:, 3] / m[:, 0] - r[:, 3] / r[:, 0]),
        "weak_time_delta_percentage_points": 100
        * (m[:, 4] / m[:, 0] - r[:, 4] / r[:, 0]),
        "short_stay_delta_percentage_points": 100
        * (m[:, 5] / m[:, 6] - r[:, 5] / r[:, 6]),
    }
    return {
        "groups": len(groups),
        "samples": samples,
        "seed": 20261030,
        "interval_95": {
            key: np.quantile(value, [0.025, 0.975]).tolist()
            for key, value in values.items()
        },
        "limitation": (
            "Conditional on these frozen policies and evaluated scenarios. "
            "The intervals do not quantify training-seed, city, or layout-population uncertainty."
        ),
    }


def load_ablation_run(run_dir: Path, expected_graph):
    run_dir = run_dir.resolve()
    complete = read_json(run_dir / "COMPLETE.json")
    if complete.get("complete") is not True:
        raise ValueError(f"Training run is not complete: {run_dir}")
    selected = complete.get("selected")
    if not isinstance(selected, dict) or not selected.get("checkpoint"):
        raise ValueError(f"No development-selected checkpoint in {run_dir}")

    checkpoint = run_dir / selected["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256(checkpoint) != selected.get("checkpoint_sha256"):
        raise ValueError(f"Selected checkpoint hash mismatch: {checkpoint}")

    config = read_json(run_dir / "run_config.json")
    settings = config["context"]["settings"]
    ablation = settings.get("ablation", {})
    implementation_dir = Path(__file__).resolve().parent
    expected_ablation = {
        "architecture": "nodewise_mlp_no_graph",
        "model_sha256": sha256(implementation_dir / "nodewise_mlp.py"),
        "launcher_sha256": sha256(implementation_dir / "train_non_gnn.py"),
    }
    matched_hash = ablation.get("matched_gat_config_sha256", "")
    if (
        any(ablation.get(key) != value for key, value in expected_ablation.items())
        or not isinstance(ablation.get("matched_gat_config"), str)
        or len(matched_hash) != 64
        or any(char not in "0123456789abcdef" for char in matched_hash)
    ):
        raise ValueError(
            f"Ablation implementation mismatch for run {run_dir}. "
            "Use the same HPCOrbit ablation code that produced the checkpoint."
        )
    seed = settings["agent_seed"]
    label = f"mlp_seed_{seed}"
    agent, _ = DQNAgent.load_policy(
        checkpoint,
        expected_graph=expected_graph,
        expected_feature_names=FEATURE_NAMES,
    )
    if not isinstance(agent.policy, NodewiseMLPQNetwork):
        raise ValueError(f"Expected an MLP policy in {checkpoint}")
    if agent.memory is not None or agent.policy.training:
        raise ValueError(f"Expected a frozen inference policy in {checkpoint}")
    return label, agent, {
        "run_dir_name": run_dir.name,
        "agent_seed": seed,
        "route_seed": settings["route_seed"],
        "ablation": ablation,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(run_dir / "run_config.json"),
    }


def archived_case(project: Path, area: str, index: int, scenario: dict, controllers: list):
    case_dir = project / "evidence" / area / "cases" / f"case_{index:04d}"
    marker = read_json(case_dir / "COMPLETE.json")
    results_path = case_dir / "results.json"
    if sha256(results_path) != marker["files"]["results.json"]:
        raise ValueError(f"Archived case hash mismatch: {case_dir}")
    if area == "heldout" and marker.get("route") != scenario["route"]:
        raise ValueError(f"Archived route mismatch: {case_dir}")
    if area == "pokhara" and marker.get("scenario") != scenario:
        raise ValueError(f"Archived scenario mismatch: {case_dir}")

    values = read_json(results_path)
    if set(values) != set(controllers):
        raise ValueError(f"Archived controller set mismatch: {case_dir}")
    if "a3" not in values:
        raise ValueError("The frozen evaluation plan must contain the a3 controller")
    for label, row in values.items():
        for key in ("route", "traffic_seed", "radio_seed"):
            if row[key] != scenario[key]:
                raise ValueError(f"Archived scenario mismatch for {label}: {key}")
        if row.get("trajectory_complete") is not True:
            raise ValueError(f"Archived trajectory is incomplete: {case_dir}")
        for key in PAIR_KEYS[3:]:
            if row[key] != values["a3"][key]:
                raise ValueError(f"Archived pairing mismatch for {label}: {key}")
    return values


def evaluate(project: Path, area: str, run_dirs: list[Path], output: Path, samples: int):
    project = project.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    install_nodewise_mlp()

    plan = read_json(project / "evidence/heldout/EVALUATION_PLAN.json")
    controllers = plan["controllers"]
    scenarios = (
        plan["scenarios"]
        if area == "heldout"
        else read_json(project / "data/pokhara/SCENARIOS.json")
    )
    run_configs = [read_json(Path(run_dir) / "run_config.json") for run_dir in run_dirs]
    first_settings = run_configs[0]["context"]["settings"]
    reference_config = first_settings.get("reference_config")
    split_config = first_settings.get("split_config")
    for config in run_configs[1:]:
        settings = config["context"]["settings"]
        if (
            settings.get("reference_config") != reference_config
            or settings.get("split_config") != split_config
        ):
            raise ValueError("Ablation runs use different reference or split configs")
    dataset = OriginalDataset(project, reference_config, split_config)
    reference_route = plan["scenarios"][0]["route"]
    original_graph = dataset.interface(reference_route).graph

    policies = {}
    run_metadata = []
    for run_dir in run_dirs:
        label, agent, metadata = load_ablation_run(run_dir, original_graph)
        if label in controllers or label in policies:
            raise ValueError(f"Duplicate controller label: {label}")
        policies[label] = agent
        run_metadata.append(metadata)

    if not policies:
        raise ValueError("Supply at least one completed non-GNN run directory")
    labels = controllers + list(policies)
    cases = []
    output_lines = []

    for index, scenario in enumerate(scenarios, start=1):
        values = archived_case(project, area, index, scenario, controllers)
        vi = (
            dataset.interface(scenario["route"], 0.875)
            if area == "heldout"
            else pokhara_interface(project, scenario)
        )
        for label, native in policies.items():
            policy = (
                native
                if area == "heldout"
                else TransferPolicy(native, vi.graph, FEATURE_NAMES)
            )
            result, *_ = rollout(
                vi,
                policy,
                scenario["traffic_seed"],
                scenario["radio_seed"],
                scenario["route"],
                counterfactual=False,
            )
            for key in ("route", "traffic_seed", "radio_seed"):
                if result[key] != scenario[key]:
                    raise RuntimeError(f"Ablation result scenario mismatch: {key}")
            if result.get("trajectory_complete") is not True:
                raise RuntimeError("Ablation produced an incomplete trajectory")
            for key in PAIR_KEYS[3:]:
                if result[key] != values["a3"][key]:
                    raise RuntimeError(f"Paired evaluation mismatch: {label}, {key}")
            values[label] = result

        cases.append(values)
        output_lines.append(
            json.dumps(
                {"case": index, "scenario": scenario, "results": values},
                allow_nan=False,
            )
        )
        if index % 25 == 0 or index == len(scenarios):
            print(f"{area}: evaluated {index}/{len(scenarios)} scenarios", flush=True)

    metrics = {
        label: aggregate([case[label] for case in cases]) for label in labels
    }
    gat_labels = [
        label for label in controllers if label != "a3" and label in plan["policies"]
    ]
    comparisons = {}
    pairwise_gat = {}
    baseline_rate = metrics["a3"]["handovers"]
    baseline_throughput = metrics["a3"]["throughput_mbps"]
    for label in labels:
        if label == "a3":
            continue
        model = metrics[label]
        deltas = [
            case[label]["mean_user_delivered_mbps"]
            - case["a3"]["mean_user_delivered_mbps"]
            for case in cases
        ]
        comparisons[label] = {
            "throughput_gain_percent": 100
            * (model["throughput_mbps"] / baseline_throughput - 1),
            "handover_reduction_percent": 100
            * (1 - model["handovers"] / baseline_rate)
            if baseline_rate
            else None,
            "scenario_wins": sum(value > 1e-9 for value in deltas),
            "scenario_losses": sum(value < -1e-9 for value in deltas),
            "scenario_ties": sum(abs(value) <= 1e-9 for value in deltas),
            "clustered_intervals": cluster_bootstrap(
                cases, scenarios, label, area, samples
            ),
        }
        if label in policies:
            pairwise_gat[label] = {}
            for gat_label in gat_labels:
                model_metrics = metrics[label]
                gat_metrics = metrics[gat_label]
                pairwise_gat[label][gat_label] = {
                    "throughput_delta_mbps": (
                        model_metrics["throughput_mbps"]
                        - gat_metrics["throughput_mbps"]
                    ),
                    "handover_rate_delta_per_hour": (
                        model_metrics["handovers_per_hour"]
                        - gat_metrics["handovers_per_hour"]
                    ),
                    "short_stay_delta_percentage_points": (
                        model_metrics["short_stay_percent"]
                        - gat_metrics["short_stay_percent"]
                    ),
                    "clustered_intervals": paired_cluster_intervals(
                        cases, scenarios, label, gat_label, area, samples
                    ),
                }

    (output / "per_case.jsonl").write_text("\n".join(output_lines) + "\n")
    write_json(
        output / "summary.json",
        {
            "scope": "Matched ablation evaluation using frozen historical baselines and new MLP policies",
            "area": area,
            "case_count": len(cases),
            "bootstrap_samples": samples,
            "metrics": metrics,
            "comparisons_vs_a3": comparisons,
            "mlp_vs_each_gat_policy": pairwise_gat,
            "policy_runs": run_metadata,
            "limitations": [
                "Intervals are conditional on these frozen policies and evaluated scenarios.",
                "The intervals do not quantify training-seed, city, or layout-population uncertainty.",
                "Historical baseline cases are reused after checking their recorded SHA-256 digests.",
            ],
        },
    )
    write_json(
        output / "COMPLETE.json",
        {
            "complete": True,
            "area": area,
            "case_count": len(cases),
            "files": {
                "per_case.jsonl": sha256(output / "per_case.jsonl"),
                "summary.json": sha256(output / "summary.json"),
            },
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--area", choices=("heldout", "pokhara"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    result = evaluate(
        args.project,
        args.area,
        args.run_dir,
        args.output,
        args.bootstrap_samples,
    )
    print(f"Completed {args.area} ablation evaluation: {result}")


if __name__ == "__main__":
    main()
