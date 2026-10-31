"""Graph-free, node-wise MLP ablation for the TopoNet DDQN agent.

The input and output interface matches GraphDuelingQNetwork. The adjacency
argument is accepted for API compatibility and is deliberately not read.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from toponet.models.graph_q import NetworkSettings


class NodewiseMLPQNetwork(nn.Module):
    """Shared per-cell MLP with the same dueling output heads as the GAT model."""

    def __init__(self, settings: NetworkSettings | None = None):
        super().__init__()
        self.settings = NetworkSettings() if settings is None else settings
        f = self.settings
        dimensions = (f.input_features, f.hidden_dim, f.head_hidden_dim)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in dimensions):
            raise ValueError("Network dimensions must be positive integers.")

        self.node1 = nn.Linear(f.input_features, f.hidden_dim)
        self.node2 = nn.Linear(f.hidden_dim, f.hidden_dim)
        for layer in (self.node1, self.node2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.norm = nn.LayerNorm(f.hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(f.hidden_dim, f.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(f.head_hidden_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(f.hidden_dim, f.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(f.head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, _graph: torch.Tensor) -> torch.Tensor:
        f = self.settings
        if (
            x.ndim != 3
            or x.shape[0] < 1
            or x.shape[1] < 1
            or x.shape[-1] != f.input_features
        ):
            raise ValueError("Features must be [B,N,input_features].")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Features must be finite floating point.")

        hidden = F.elu(self.node1(x))
        hidden = self.norm(hidden + F.elu(self.node2(hidden)))
        value = self.value_head(hidden.mean(dim=1))
        advantage = self.advantage_head(hidden).squeeze(-1)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def install_nodewise_mlp() -> None:
    """Route DQNAgent construction and checkpoint loading through this model."""
    import toponet.agents.ddqn as ddqn

    if ddqn.GraphDuelingQNetwork is NodewiseMLPQNetwork:
        return
    from toponet.models.graph_q import GraphDuelingQNetwork

    if ddqn.GraphDuelingQNetwork is not GraphDuelingQNetwork:
        raise RuntimeError("Unexpected Q-network replacement is already installed.")
    ddqn.GraphDuelingQNetwork = NodewiseMLPQNetwork
