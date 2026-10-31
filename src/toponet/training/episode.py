"""Training / episode.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import torch
from toponet.envs.features import LearningInterface, FEATURE_NAMES
from toponet.data.input_inspection import load_environment_inputs
from toponet.agents.ddqn import DQNAgent, AgentSettings


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(tree_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) == type(b)
            and len(a) == len(b)
            and all(tree_equal(x, y) for x, y in zip(a, b))
        )
    return a == b


def learning_snapshot(agent):
    return {
        "policy": {k: v.detach().clone() for k, v in agent.policy.state_dict().items()},
        "target": {k: v.detach().clone() for k, v in agent.target.state_dict().items()},
        "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
        "action_rng": copy.deepcopy(agent.action_rng.bit_generator.state),
        "replay_rng": copy.deepcopy(agent.memory.rng.bit_generator.state)
        if agent.memory is not None
        else None,
        "environment_steps": agent.environment_steps,
        "optimizer_updates": agent.optimizer_updates,
        "target_syncs": agent.target_syncs,
        "replay_size": len(agent.memory) if agent.memory is not None else 0,
    }


def run_episode(
    interface,
    agent,
    traffic_seed,
    radio_seed,
    training,
    max_steps=None,
    spatial_mode="nominal",
):
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive.")
    snapshot = None if training else learning_snapshot(agent)
    state = interface.reset(traffic_seed, radio_seed, spatial_mode)
    steps = 0
    losses = []
    started = time.perf_counter()
    while True:
        action = agent.select_action(state, eval_mode=not training)
        next_state, reward, done, info = interface.step(action)
        if training:
            update = agent.observe_transition(
                state, action, reward, next_state, done, info["duration_s"]
            )
            if update is not None:
                losses.append(update["loss"])
        state = next_state
        steps += 1
        if done or (max_steps is not None and steps >= max_steps):
            break
    if snapshot is not None and not tree_equal(snapshot, learning_snapshot(agent)):
        raise RuntimeError("Validation altered learning state.")
    # A smoke budget cutoff is a truncation, NEVER a false terminal DDQN target.
    result = {
        **interface.env.summary(),
        "steps": steps,
        "trajectory_complete": bool(done),
        "budget_truncated": not bool(done),
        "training": training,
        "spatial_mode": spatial_mode,
        "traffic_seed": traffic_seed,
        "radio_seed": radio_seed,
        "reward_total": interface.reward_total,
        "reward_per_s": interface.reward_total / interface.env.elapsed_s,
        "reward_components": dict(interface.reward_totals),
        "wall_time_s": time.perf_counter() - started,
        "mean_update_loss": float(np.mean(losses)) if losses else None,
        "epsilon": agent.epsilon,
        "optimizer_updates": agent.optimizer_updates,
    }
    return result, state


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()
