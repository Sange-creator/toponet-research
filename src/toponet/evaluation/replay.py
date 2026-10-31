"""Replay explicitly requested frozen cases; historical evidence stays read-only."""

from pathlib import Path
import torch
from toponet.agents.ddqn import DQNAgent
from toponet.envs.features import FEATURE_NAMES
from toponet.data.datasets import OriginalDataset, pokhara_interface, read_json
from toponet.baselines.controllers import A3StyleController, LoadAwareController
from toponet.evaluation.transfer import TransferPolicy
from toponet.evaluation.diagnostics import rollout
from toponet.io import sha256, write_json


def replay(project, output, area="heldout", index=1, controllers=None):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new replay output directory")
    torch.set_num_threads(1)
    dataset = OriginalDataset(project)
    plan = read_json(project / "evidence/heldout/EVALUATION_PLAN.json")
    scenarios = (
        plan["scenarios"]
        if area == "heldout"
        else read_json(project / "data/pokhara/SCENARIOS.json")
    )
    if not 1 <= index <= len(scenarios):
        raise ValueError("Case index out of range")
    scenario = scenarios[index - 1]
    labels = controllers or plan["controllers"]
    if len(labels) != len(set(labels)) or not set(labels) <= set(
        plan["controllers"] + ["load_aware"]
    ):
        raise ValueError("Unknown or repeated controller")
    output.mkdir(parents=True)
    results = {}
    for label in labels:
        vi = (
            dataset.interface(scenario["route"], 0.875)
            if area == "heldout"
            else pokhara_interface(project, scenario)
        )
        agent = None
        controller = (
            A3StyleController()
            if label == "a3"
            else LoadAwareController()
            if label == "load_aware"
            else None
        )
        if controller is None:
            policy = plan["policies"][label]
            path = project / "checkpoints/frozen" / Path(policy["frozen_copy"]).name
            if sha256(path) != policy["sha256"]:
                raise ValueError("Frozen checkpoint changed")
            original_graph = dataset.interface(plan["scenarios"][0]["route"]).graph
            agent, _ = DQNAgent.load_policy(
                path,
                expected_graph=original_graph,
                expected_feature_names=FEATURE_NAMES,
            )
            if area == "pokhara":
                agent = TransferPolicy(agent, vi.graph, FEATURE_NAMES)
        result, *_ = rollout(
            vi,
            agent,
            scenario["traffic_seed"],
            scenario["radio_seed"],
            scenario["route"],
            controller,
            counterfactual=False,
        )
        if results:
            reference = next(iter(results.values()))
            for key in (
                "route",
                "traffic_seed",
                "radio_seed",
                "steps",
                "elapsed_s",
                "initial_serving_site_id",
                "external_state_sha256",
            ):
                if result[key] != reference[key]:
                    raise RuntimeError("Replay pairing failed: " + key)
        results[label] = result
        write_json(output / "results.json", results)
    write_json(
        output / "COMPLETE.json",
        dict(
            complete=True,
            area=area,
            scenario=scenario,
            controllers=labels,
            results_sha256=sha256(output / "results.json"),
            scope="New replay; not a new independent sample. Load-aware defaults, if requested, were not tuned.",
        ),
    )
    return dict(output=str(output), area=area, case=index, controllers=labels)
