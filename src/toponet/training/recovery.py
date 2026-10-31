"""Training / recovery.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from pathlib import Path
from dataclasses import asdict
import copy, json, os, random
import numpy as np
import torch

ARRAYS = (
    "states",
    "next_states",
    "masks",
    "next_masks",
    "actions",
    "rewards",
    "dones",
    "durations",
)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def snapshot(agent, sampler, progress, context, state):
    m = agent.memory
    return dict(
        format="toponet_boundary_recovery_v1",
        context=context,
        agent_settings=asdict(agent.settings),
        network_settings=asdict(agent.network_settings),
        seed=agent.seed,
        graph=torch.from_numpy(agent.graph),
        policy=agent.policy.state_dict(),
        target=agent.target.state_dict(),
        optimizer=agent.optimizer.state_dict(),
        policy_training=agent.policy.training,
        target_training=agent.target.training,
        environment_steps=agent.environment_steps,
        optimizer_updates=agent.optimizer_updates,
        target_syncs=agent.target_syncs,
        action_rng=copy.deepcopy(agent.action_rng.bit_generator.state),
        replay_rng=copy.deepcopy(m.rng.bit_generator.state),
        replay_size=m.size,
        replay_cursor=m.cursor,
        replay={k: torch.from_numpy(getattr(m, k)[: m.size]) for k in ARRAYS},
        sampler=dict(
            rng=sampler.rng.getstate(),
            start_bag=sampler.start_bag.copy(),
            route_bags=copy.deepcopy(sampler.route_bags),
            by_start=copy.deepcopy(sampler.by_start),
        ),
        python_rng=random.getstate(),
        numpy_rng=dict(
            name=np.random.get_state()[0],
            keys=torch.from_numpy(np.random.get_state()[1].astype(np.int64)),
            position=np.random.get_state()[2],
            has_gauss=np.random.get_state()[3],
            cached_gaussian=np.random.get_state()[4],
        ),
        torch_rng=torch.get_rng_state(),
        progress=copy.deepcopy(progress),
        last_state=None
        if state is None
        else {k: torch.from_numpy(np.array(v, copy=True)) for k, v in state.items()},
    )


def save_boundary(folder, agent, sampler, progress, context, state):
    folder = Path(folder)
    latest = folder / "recovery_latest.pt"
    previous = folder / "recovery_previous.pt"
    tmp = folder / "recovery_writing.tmp"
    payload = snapshot(agent, sampler, progress, context, state)
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    if latest.exists():
        os.replace(latest, previous)
    os.replace(tmp, latest)
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    atomic_json(
        folder / "RECOVERY_STATUS.json",
        dict(
            completed_episodes=progress["completed"],
            environment_steps=agent.environment_steps,
            optimizer_updates=agent.optimizer_updates,
            checkpoint="recovery_latest.pt",
            boundary="completed journey or initial state",
        ),
    )


def read_boundary(folder, context):
    errors = []
    for n in ("recovery_latest.pt", "recovery_previous.pt"):
        path = Path(folder) / n
        if not path.exists():
            continue
        try:
            p = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            errors.append(n + ": " + str(e))
            continue
        if (
            p.get("format") != "toponet_boundary_recovery_v1"
            or p.get("context") != context
        ):
            raise ValueError(
                "Recovery protocol/source/settings mismatch; do not bypass"
            )
        return p, n
    raise RuntimeError("No readable boundary checkpoint. " + str(errors))


def restore(agent, sampler, p):
    if (
        agent.device.type != "cpu"
        or agent.seed != p["seed"]
        or asdict(agent.settings) != p["agent_settings"]
        or asdict(agent.network_settings) != p["network_settings"]
        or not np.array_equal(agent.graph, p["graph"].numpy())
    ):
        raise ValueError("Agent checkpoint mismatch")
    m = agent.memory
    n = p["replay_size"]
    cursor = p["replay_cursor"]
    if (
        not 0 <= n <= m.capacity
        or not 0 <= cursor < m.capacity
        or (n < m.capacity and cursor != n)
    ):
        raise ValueError("Invalid replay counters")
    if sampler.by_start != p["sampler"]["by_start"]:
        raise ValueError("Sampler training split mismatch")
    for k in ARRAYS:
        x = p["replay"][k].numpy()
        dest = getattr(m, k)
        if x.shape != dest[:n].shape or x.dtype != dest.dtype:
            raise ValueError("Replay shape/dtype mismatch")
        dest[:n] = x
    m.size = n
    m.cursor = cursor
    m.rng.bit_generator.state = p["replay_rng"]
    agent.action_rng.bit_generator.state = p["action_rng"]
    agent.policy.load_state_dict(p["policy"])
    agent.target.load_state_dict(p["target"])
    agent.optimizer.load_state_dict(p["optimizer"])
    agent.policy.train(p["policy_training"])
    agent.target.train(p["target_training"])
    for k in ("environment_steps", "optimizer_updates", "target_syncs"):
        setattr(agent, k, p[k])
    sampler.rng.setstate(p["sampler"]["rng"])
    sampler.start_bag = p["sampler"]["start_bag"].copy()
    sampler.route_bags = copy.deepcopy(p["sampler"]["route_bags"])
    random.setstate(p["python_rng"])
    r = p["numpy_rng"]
    np.random.set_state(
        (
            r["name"],
            r["keys"].numpy().astype(np.uint32),
            r["position"],
            r["has_gauss"],
            r["cached_gaussian"],
        )
    )
    torch.set_rng_state(p["torch_rng"])
    state = (
        None
        if p["last_state"] is None
        else {k: v.numpy().copy() for k, v in p["last_state"].items()}
    )
    return copy.deepcopy(p["progress"]), state
