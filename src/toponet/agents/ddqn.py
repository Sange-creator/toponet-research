"""Agents / ddqn.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import math
import tempfile
import numpy as np
import torch
from torch import nn
from toponet.models.graph_q import (
    GraphDuelingQNetwork,
    NetworkSettings,
    masked_q_values,
)


@dataclass(frozen=True)
class AgentSettings:
    learning_rate: float = 1e-3
    gamma_reference_step: float = 0.99
    reference_step_s: float = 0.5
    replay_capacity: int = 20000
    batch_size: int = 32
    warmup_steps: int = 2000
    update_every_steps: int = 4
    target_every_updates: int = 1000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 100000
    gradient_clip_norm: float = 10.0


def validate_settings(f):
    for k in (
        "replay_capacity",
        "batch_size",
        "update_every_steps",
        "target_every_updates",
        "epsilon_decay_steps",
    ):
        v = getattr(f, k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"Invalid {k}.")
    if (
        isinstance(f.warmup_steps, bool)
        or not isinstance(f.warmup_steps, int)
        or f.warmup_steps < 0
    ):
        raise ValueError("Invalid warmup.")
    if f.batch_size > f.replay_capacity:
        raise ValueError("Batch exceeds replay capacity.")
    vals = (
        f.learning_rate,
        f.gamma_reference_step,
        f.reference_step_s,
        f.epsilon_start,
        f.epsilon_final,
        f.gradient_clip_norm,
    )
    if (
        not all(math.isfinite(v) for v in vals)
        or f.learning_rate <= 0
        or not 0 < f.gamma_reference_step <= 1
        or f.reference_step_s <= 0
        or not 0 <= f.epsilon_final <= f.epsilon_start <= 1
        or f.gradient_clip_norm <= 0
    ):
        raise ValueError("Invalid agent scales.")


def checked_state(state, graph, input_features, allow_empty_mask=False):
    x = np.asarray(state["node_features"], dtype=np.float32)
    mask = np.asarray(state["action_mask"])
    g = np.asarray(state["message_graph"])
    n = graph.shape[0]
    if x.shape != (n, input_features) or not np.isfinite(x).all():
        raise ValueError("Invalid node features.")
    if (
        mask.dtype != np.bool_
        or mask.shape != (n,)
        or (not allow_empty_mask and not mask.any())
    ):
        raise ValueError("Invalid action mask.")
    if g.shape != graph.shape or not np.array_equal(g, graph):
        raise ValueError("Replay agent requires the same fixed message graph.")
    return x, mask


class ReplayBuffer:
    """Copy-on-insertion ring; graph is stored once, not copied per transition."""

    def __init__(self, capacity, graph, input_features, seed=7):
        self.capacity = capacity
        self.graph = np.array(graph, dtype=np.float32, copy=True)
        self.input_features = input_features
        self.size = 0
        self.cursor = 0
        self.rng = np.random.default_rng(seed)
        n = len(graph)
        self.states = np.empty((capacity, n, input_features), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.masks = np.empty((capacity, n), dtype=bool)
        self.next_masks = np.empty_like(self.masks)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=bool)
        self.durations = np.empty(capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, done, duration_s):
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("Done must be bool.")
        x, mask = checked_state(state, self.graph, self.input_features)
        nx, nmask = checked_state(
            next_state, self.graph, self.input_features, allow_empty_mask=done
        )
        if (
            isinstance(action, bool)
            or not isinstance(action, (int, np.integer))
            or not 0 <= action < len(mask)
            or not mask[action]
        ):
            raise ValueError("Replay action was not legal.")
        if (
            not math.isfinite(reward)
            or abs(reward) > np.finfo(np.float32).max
            or not math.isfinite(duration_s)
            or duration_s <= 0
            or duration_s > np.finfo(np.float32).max
        ):
            raise ValueError("Invalid transition reward/duration.")
        i = self.cursor
        self.states[i] = x
        self.next_states[i] = nx
        self.masks[i] = mask
        self.next_masks[i] = nmask
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.durations[i] = duration_s
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("Not enough replay transitions.")
        indices = self.rng.choice(self.size, batch_size, replace=False)
        arrays = {
            "states": self.states[indices],
            "next_states": self.next_states[indices],
            "masks": self.masks[indices],
            "next_masks": self.next_masks[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "durations": self.durations[indices],
        }
        return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


@torch.no_grad()
def double_dqn_targets(
    policy,
    target,
    next_states,
    next_masks,
    rewards,
    dones,
    durations,
    graph,
    gamma_reference_step=0.99,
    reference_step_s=0.5,
):
    """Select legal next action online; evaluate that SAME action using target.
    Terminal rows do not enter max/gather, avoiding 0 * (-inf) or invalid masks.
    Discount gamma^(duration/reference_dt) handles shorter final intervals.
    """
    result = rewards.clone()
    live = ~dones
    if live.any():
        next_graph = graph[live] if graph.ndim == 3 else graph
        online_q = policy(next_states[live], next_graph)
        actions = masked_q_values(online_q, next_masks[live]).argmax(
            dim=1, keepdim=True
        )
        target_q = target(next_states[live], next_graph).gather(1, actions).squeeze(1)
        discount = torch.pow(
            torch.full_like(durations[live], gamma_reference_step),
            durations[live] / reference_step_s,
        )
        result[live] += discount * target_q
    if not torch.isfinite(result).all():
        raise RuntimeError("Nonfinite DDQN target.")
    return result


class DQNAgent:
    def __init__(
        self,
        graph,
        input_features=15,
        seed=7,
        device="cpu",
        settings=None,
        network_settings=None,
        allocate_replay=True,
    ):
        self.settings = AgentSettings() if settings is None else settings
        validate_settings(self.settings)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Seed must be nonnegative integer.")
        self.seed = seed
        self.device = torch.device(device)
        self.graph = np.array(graph, dtype=np.float32, copy=True)
        if (
            self.graph.ndim != 2
            or self.graph.shape[0] < 1
            or self.graph.shape[0] != self.graph.shape[1]
            or not np.isfinite(self.graph).all()
            or (self.graph < 0).any()
            or not (self.graph > 0).any(axis=1).all()
        ):
            raise ValueError("Invalid message graph.")
        self.network_settings = (
            NetworkSettings(input_features=input_features)
            if network_settings is None
            else network_settings
        )
        if self.network_settings.input_features != input_features:
            raise ValueError("Feature dimension mismatch.")
        self.input_features = input_features
        self.graph_tensor = torch.tensor(self.graph, device=self.device)
        devices = (
            []
            if self.device.type != "cuda"
            else [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.policy = GraphDuelingQNetwork(self.network_settings).to(self.device)
        self.target = copy.deepcopy(self.policy).to(self.device).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.settings.learning_rate
        )
        action_seed, replay_seed = np.random.SeedSequence(seed).spawn(2)
        self.action_rng = np.random.default_rng(action_seed)
        self.memory = (
            ReplayBuffer(
                self.settings.replay_capacity, self.graph, input_features, replay_seed
            )
            if allocate_replay
            else None
        )
        self.environment_steps = 0
        self.optimizer_updates = 0
        self.target_syncs = 0

    @property
    def epsilon(self):
        f = self.settings
        progress = np.clip(
            (self.environment_steps - f.warmup_steps) / f.epsilon_decay_steps, 0.0, 1.0
        )
        return float(f.epsilon_start + (f.epsilon_final - f.epsilon_start) * progress)

    def select_action(self, state, eval_mode=False):
        x, mask = checked_state(state, self.graph, self.input_features)
        if not eval_mode and self.action_rng.random() < self.epsilon:
            return int(self.action_rng.choice(np.flatnonzero(mask)))
        was_training = self.policy.training
        self.policy.eval()
        try:
            with torch.no_grad():
                q = self.policy(
                    torch.tensor(x, device=self.device).unsqueeze(0), self.graph_tensor
                )
                legal = masked_q_values(q, torch.tensor(mask, device=self.device))
                return int(legal.argmax(1).item())
        finally:
            self.policy.train(was_training)

    def observe_transition(self, state, action, reward, next_state, done, duration_s):
        if self.memory is None:
            raise RuntimeError("Frozen policy has no replay. Do not train it.")
        self.memory.push(state, action, reward, next_state, done, duration_s)
        self.environment_steps += 1
        f = self.settings
        if (
            self.environment_steps >= f.warmup_steps
            and self.environment_steps % f.update_every_steps == 0
            and len(self.memory) >= f.batch_size
        ):
            return self.update()
        return None

    def update(self):
        if self.memory is None or len(self.memory) < self.settings.batch_size:
            return None
        batch = self.memory.sample(self.settings.batch_size, self.device)
        self.policy.train()
        current = (
            self.policy(batch["states"], self.graph_tensor)
            .gather(1, batch["actions"].unsqueeze(1))
            .squeeze(1)
        )
        expected = double_dqn_targets(
            self.policy,
            self.target,
            batch["next_states"],
            batch["next_masks"],
            batch["rewards"],
            batch["dones"],
            batch["durations"],
            self.graph_tensor,
            self.settings.gamma_reference_step,
            self.settings.reference_step_s,
        )
        loss = nn.functional.smooth_l1_loss(current, expected)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite DQN loss.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.settings.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.optimizer_updates += 1
        if self.optimizer_updates % self.settings.target_every_updates == 0:
            self.target.load_state_dict(self.policy.state_dict())
            self.target_syncs += 1
        return {
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(norm),
            "updates": self.optimizer_updates,
            "epsilon": self.epsilon,
        }

    def save_policy(self, path, metadata):
        """Use trusted local checkpoints only. JSON metadata must capture experiment.
        Original model+optimizer+RNG+replay are NOT persisted; no exact resume claim.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError("Checkpoint already exists; use a new name.")
        json.dumps(metadata, allow_nan=False)
        feature_names = metadata.get("feature_names")
        if (
            not isinstance(feature_names, list)
            or len(feature_names) != self.input_features
            or len(set(feature_names)) != len(feature_names)
        ):
            raise ValueError("Provide ordered unique feature_names in metadata.")
        payload = {
            "format_version": 1,
            "purpose": "frozen_policy_not_full_resume",
            "network_settings": asdict(self.network_settings),
            "agent_settings": asdict(self.settings),
            "seed": self.seed,
            "graph": torch.tensor(self.graph),
            "policy_state": {
                k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()
            },
            "environment_steps": self.environment_steps,
            "optimizer_updates": self.optimizer_updates,
            "metadata": metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load_policy(
        cls, path, device="cpu", expected_graph=None, expected_feature_names=None
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format_version") != 1
            or payload.get("purpose") != "frozen_policy_not_full_resume"
        ):
            raise ValueError("Unsupported checkpoint.")
        graph = payload["graph"].numpy()
        metadata = payload["metadata"]
        if expected_graph is not None and not np.array_equal(graph, expected_graph):
            raise ValueError("Checkpoint graph mismatch.")
        if expected_feature_names is not None and list(
            expected_feature_names
        ) != metadata.get("feature_names"):
            raise ValueError("Checkpoint feature schema mismatch.")
        network_settings = NetworkSettings(**payload["network_settings"])
        agent = cls(
            graph,
            network_settings.input_features,
            payload["seed"],
            device,
            AgentSettings(**payload["agent_settings"]),
            network_settings,
            allocate_replay=False,
        )
        agent.policy.load_state_dict(payload["policy_state"])
        agent.policy.eval()
        agent.target.load_state_dict(agent.policy.state_dict())
        agent.environment_steps = payload["environment_steps"]
        agent.optimizer_updates = payload["optimizer_updates"]
        return agent, metadata


def self_test():
    torch.set_num_threads(1)
    graph = np.eye(3, dtype=np.float32)

    def state(mask=(True, False, True)):
        return {
            "node_features": np.full((3, 15), 0.3, dtype=np.float32),
            "action_mask": np.array(mask, dtype=bool),
            "message_graph": graph.copy(),
        }

    s = state()
    ns = state()
    replay = ReplayBuffer(2, graph, 15, 7)
    replay.push(s, 0, 1.0, ns, False, 0.5)
    s["node_features"][:] = 0.0
    assert (replay.states[0] == 0.3).all()
    for _ in range(3):
        replay.push(state(), 0, 2.0, state(), False, 0.5)
    assert len(replay) == 2 and replay.cursor == 0
    try:
        replay.push(state(), 1, 0.0, state(), False, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Illegal action accepted into replay.")

    # Controlled scores: online illegal action is huge; target's own best is not
    # the online-selected legal action. DDQN must choose node 2, evaluate value 7.
    class Scores(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.register_buffer("values", torch.tensor(v, dtype=torch.float32))

        def forward(self, x, g):
            return self.values.expand(len(x), -1)

    x = torch.zeros(3, 3, 15)
    m = torch.tensor([[True, False, True], [False, False, False], [True, False, True]])
    y = double_dqn_targets(
        Scores([1.0, 100.0, 3.0]),
        Scores([10.0, 1000.0, 7.0]),
        x,
        m,
        torch.tensor([1.0, 2.0, 1.0]),
        torch.tensor([False, True, False]),
        torch.tensor([0.5, 0.2, 0.25]),
        torch.eye(3),
    )
    assert torch.allclose(
        y, torch.tensor([1.0 + 0.99 * 7.0, 2.0, 1.0 + math.sqrt(0.99) * 7.0])
    )
    f = AgentSettings(
        replay_capacity=16,
        batch_size=4,
        warmup_steps=4,
        update_every_steps=1,
        target_every_updates=2,
        epsilon_decay_steps=10,
    )
    a = DQNAgent(graph, settings=f)
    b = DQNAgent(graph, settings=f)
    assert all(
        torch.equal(a.policy.state_dict()[k], b.policy.state_dict()[k])
        for k in a.policy.state_dict()
    )
    assert a.epsilon == 1.0
    for _ in range(100):
        assert a.select_action(state()) in (0, 2)
    initial = {k: v.clone() for k, v in a.policy.state_dict().items()}
    for _ in range(6):
        a.observe_transition(state(), 0, 1.0, state(), False, 0.5)
    assert a.optimizer_updates == 3 and a.target_syncs == 1
    assert any(not torch.equal(v, initial[k]) for k, v in a.policy.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in a.target.parameters())
    a.environment_steps = 14
    assert np.isclose(a.epsilon, 0.05)
    # Greedy evaluation changes no weights, replay, counter or exploration RNG.
    rng = copy.deepcopy(a.action_rng.bit_generator.state)
    steps = a.environment_steps
    updates = a.optimizer_updates
    size = len(a.memory)
    greedy = a.select_action(state(), eval_mode=True)
    assert (
        greedy in (0, 2)
        and a.action_rng.bit_generator.state == rng
        and a.environment_steps == steps
        and a.optimizer_updates == updates
        and len(a.memory) == size
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.pt"
        names = [f"f{i}" for i in range(15)]
        a.save_policy(path, {"feature_names": names, "test_only": True})
        restored, meta = DQNAgent.load_policy(
            path, expected_graph=graph, expected_feature_names=names
        )
        assert (
            restored.memory is None
            and restored.select_action(state(), eval_mode=True) == greedy
        )
        assert all(
            torch.equal(a.policy.state_dict()[k], restored.policy.state_dict()[k])
            for k in a.policy.state_dict()
        )
        try:
            DQNAgent.load_policy(path, expected_feature_names=list(reversed(names)))
        except ValueError:
            pass
        else:
            raise AssertionError("Schema mismatch accepted.")
    print(
        "Agent tests passed: copied/ring replay, legal exploration, masked Double-DQN targets, terminal rows, duration discount, updates, target sync, frozen evaluation and policy checkpoint round-trip."
    )
