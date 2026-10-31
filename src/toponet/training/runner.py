"""New, isolated training runs with complete-journey boundary recovery.
This driver does not overwrite or resume the historical HPCOrbit experiment.
"""

import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from toponet.agents.ddqn import DQNAgent, AgentSettings
from toponet.models.graph_q import NetworkSettings
from toponet.envs.features import FEATURE_NAMES
from toponet.data.datasets import OriginalDataset
from toponet.training.sampler import BalancedRouteSampler
from toponet.training.schedule import plan_complete_journeys
from toponet.training.recovery import save_boundary, read_boundary, restore
from toponet.training.episode import run_episode
from toponet.evaluation.diagnostics import rollout
from toponet.evaluation.metrics import aggregate, assessment, selection_rank
from toponet.baselines.controllers import A3StyleController
from toponet.io import sha256, write_json


def _train_impl(project, config_path, run_dir, resume=False, smoke=False):
    project, config_path, run_dir = map(
        lambda p: Path(p).resolve(), (project, config_path, run_dir)
    )
    settings = json.loads(config_path.read_text())
    episodes, seed = settings["episodes"], settings["agent_seed"]
    if not isinstance(episodes, int) or not 1 <= episodes <= 10000:
        raise ValueError("episodes must be 1..10000")
    torch.set_num_threads(settings.get("threads", 1))
    dataset = OriginalDataset(
        project, settings.get("reference_config"), settings.get("split_config")
    )
    config = dataset.config
    cost = settings["handover_cost"]
    plan = plan_complete_journeys(
        dataset.rows,
        dataset.split["train_route_ids"],
        episodes,
        settings["route_seed"],
        config["speed_mps"],
        config["dt_s"],
    )
    budget = plan["total_environment_steps"]
    gates = sorted(
        {
            next(
                j["cumulative_steps"]
                for j in plan["journeys"]
                if j["cumulative_steps"] >= min(target, budget)
            )
            for target in settings["validation_target_steps"] + [budget]
        }
    )
    agent_cfg = dict(config["agent_settings"], **settings.get("agent_overrides", {}))
    network_cfg = dict(
        config["network_settings"], **settings.get("network_overrides", {})
    )
    context = dict(
        protocol="toponet_clean_training_v1",
        settings=settings,
        reference_sha256=sha256(dataset.reference_path),
        split_sha256=sha256(dataset.split_path),
        source_hashes={
            str(p.relative_to(project)): sha256(p)
            for p in sorted((project / "src/toponet").rglob("*.py"))
        },
        torch=str(torch.__version__),
        numpy=np.__version__,
        python=platform.python_version(),
        smoke=smoke,
    )
    if resume and smoke:
        raise ValueError("Smoke checks do not support recovery")
    if resume:
        if (
            not (run_dir / "run_config.json").is_file()
            or json.loads((run_dir / "run_config.json").read_text())["context"]
            != context
        ):
            raise ValueError(
                "Resume requires identical clean-project settings, sources and runtime"
            )
        if (run_dir / "COMPLETE.json").exists():
            raise ValueError("Run already complete; choose a new run directory")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            run_dir / "run_config.json",
            dict(
                context=context,
                episodes=episodes,
                environment_step_budget=budget,
                validation_step_gates=gates,
                agent_settings=agent_cfg,
                network_settings=network_cfg,
            ),
        )
        write_json(run_dir / "planned_journeys.json", plan)
    first = dataset.interface(plan["journeys"][0]["route"], cost)
    agent = DQNAgent(
        first.graph,
        len(FEATURE_NAMES),
        seed,
        "cpu",
        AgentSettings(**agent_cfg),
        NetworkSettings(**network_cfg),
    )
    if smoke:
        result, state = run_episode(first, agent, 10000, 110000, True, max_steps=80)
        agent.save_policy(
            run_dir / "smoke_policy.pt",
            dict(
                feature_names=list(FEATURE_NAMES),
                scope="80-step pipeline check, not a trained-policy result",
            ),
        )
        write_json(
            run_dir / "COMPLETE.json",
            dict(
                scope="Smoke only; production warmup retained; not performance evidence",
                result=result,
            ),
        )
        return dict(
            run_dir=str(run_dir),
            steps=agent.environment_steps,
            updates=agent.optimizer_updates,
            scope="smoke_only",
        )
    sampler = BalancedRouteSampler(
        [dataset.rows[r] for r in dataset.split["train_route_ids"]],
        settings["route_seed"],
    )
    progress = dict(completed=0, journeys=[], validations=[], training_metrics=[])
    state = None
    if resume:
        saved, name = read_boundary(run_dir, context)
        progress, state = restore(agent, sampler, saved)
        del saved
        if name != "recovery_latest.pt":
            raise RuntimeError(
                "Only previous recovery readable. Preserve the corrupt latest and inspect before retrying; no automatic overwrite."
            )
    else:
        save_boundary(run_dir, agent, sampler, progress, context, state)
    # Baseline is fixed and evaluated only on the six development routes.
    validation_ids = dataset.split["validation_route_ids"]
    baseline_path = run_dir / "baseline_development.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
    else:
        rows = [
            rollout(
                dataset.interface(r, cost),
                None,
                20000 + i,
                120000 + i,
                r,
                A3StyleController(),
                False,
            )[0]
            for i, r in enumerate(validation_ids)
        ]
        baseline = dict(cases=rows, metrics=aggregate(rows))
        write_json(baseline_path, baseline)
    for index in range(progress["completed"], episodes):
        item = plan["journeys"][index]
        route = sampler.next_route_id()
        if route != item["route"]:
            raise RuntimeError("Sampler differs from declared plan")
        result, state = run_episode(
            dataset.interface(route, cost), agent, 10000 + index, 110000 + index, True
        )
        if (
            not result["trajectory_complete"]
            or result["steps"] != item["steps"]
            or agent.environment_steps != item["cumulative_steps"]
        ):
            raise RuntimeError("Complete journey accounting mismatch")
        if agent.environment_steps in gates:
            rows = [
                rollout(
                    dataset.interface(r, cost),
                    agent,
                    20000 + i,
                    120000 + i,
                    r,
                    counterfactual=False,
                )[0]
                for i, r in enumerate(validation_ids)
            ]
            for actual, reference in zip(rows, baseline["cases"]):
                for key in (
                    "route",
                    "traffic_seed",
                    "radio_seed",
                    "steps",
                    "elapsed_s",
                    "initial_serving_site_id",
                    "external_state_sha256",
                ):
                    if actual[key] != reference[key]:
                        raise RuntimeError("Development pairing mismatch: " + key)
            metrics = aggregate(rows)
            checkpoint = f"policy_step_{agent.environment_steps:08d}.pt"
            path = run_dir / checkpoint
            # An interruption may leave an uncommitted policy; never overwrite it.
            if path.exists():
                checkpoint = (
                    f"policy_step_{agent.environment_steps:08d}_{time.time_ns()}.pt"
                )
                path = run_dir / checkpoint
            agent.save_policy(
                path,
                dict(
                    feature_names=list(FEATURE_NAMES),
                    clean_training_context=context,
                    after_training_steps=agent.environment_steps,
                ),
            )
            progress["validations"].append(
                dict(
                    step=agent.environment_steps,
                    metrics=metrics,
                    assessment=assessment(metrics, baseline["metrics"]),
                    checkpoint=checkpoint,
                    checkpoint_sha256=sha256(path),
                    cases=rows,
                )
            )
        progress["completed"] = index + 1
        progress["journeys"].append(
            dict(
                route=route, steps=result["steps"], total_steps=agent.environment_steps
            )
        )
        progress["training_metrics"].append(
            dict(
                result,
                episode=index + 1,
                route=route,
                total_steps=agent.environment_steps,
            )
        )
        save_boundary(run_dir, agent, sampler, progress, context, state)
        for filename, records in [
            ("training.jsonl", progress["training_metrics"]),
            ("validation.jsonl", progress["validations"]),
        ]:
            temporary = run_dir / (filename + ".tmp")
            temporary.write_text(
                "".join(
                    json.dumps(record, allow_nan=False) + "\n" for record in records
                )
            )
            temporary.replace(run_dir / filename)
        chosen = (
            min(progress["validations"], key=selection_rank)
            if progress["validations"]
            else None
        )
        write_json(
            run_dir / "progress.json",
            dict(
                completed_episodes=progress["completed"],
                target_episodes=episodes,
                environment_steps=agent.environment_steps,
                optimizer_updates=agent.optimizer_updates,
                selected=chosen,
                validations=progress["validations"],
            ),
        )
        print(
            f"Episode {index + 1}/{episodes}; steps {agent.environment_steps}/{budget}; recovery saved",
            flush=True,
        )
    first_update = (
        math.ceil(
            max(1, agent.settings.warmup_steps, agent.settings.batch_size)
            / agent.settings.update_every_steps
        )
        * agent.settings.update_every_steps
    )
    expected_updates = max(
        0, (budget - first_update) // agent.settings.update_every_steps + 1
    )
    if (
        agent.optimizer_updates != expected_updates
        or agent.target_syncs != expected_updates // agent.settings.target_every_updates
    ):
        raise RuntimeError("Update/synchronization count mismatch")
    write_json(
        run_dir / "COMPLETE.json",
        dict(
            complete=True,
            episodes=episodes,
            steps=agent.environment_steps,
            optimizer_updates=agent.optimizer_updates,
            selected=chosen,
            scope="New development run; not automatically a replacement for frozen historical policies",
        ),
    )
    return dict(
        run_dir=str(run_dir),
        steps=agent.environment_steps,
        updates=agent.optimizer_updates,
    )


def train(project, config_path, run_dir, resume=False, smoke=False):
    """Lock a new-run name to prevent two managers writing the same recovery files."""
    import fcntl

    run_dir = Path(run_dir).resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.parent / ("." + run_dir.name + ".manager.lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another training manager already owns this run"
            ) from exc
        return _train_impl(project, config_path, run_dir, resume, smoke)
