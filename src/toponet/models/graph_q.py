"""Models / graph q.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NetworkSettings:
    input_features: int = 15
    hidden_dim: int = 64
    attention_heads: int = 4
    head_hidden_dim: int = 32
    negative_slope: float = 0.2


def graph_mask(graph, batch_size, num_nodes, device):
    if not isinstance(graph, torch.Tensor):
        raise TypeError("Graph must be a torch tensor.")
    if graph.device != device:
        raise ValueError("Graph and features must be on the same device.")
    if graph.ndim == 2 and tuple(graph.shape) == (num_nodes, num_nodes):
        graph = graph.unsqueeze(0).expand(batch_size, -1, -1)
    elif graph.ndim != 3 or tuple(graph.shape) != (batch_size, num_nodes, num_nodes):
        raise ValueError("Graph must be [N,N] or [B,N,N].")
    if not torch.isfinite(graph).all() or (graph < 0).any():
        raise ValueError("Graph entries must be finite and nonnegative.")
    mask = graph > 0
    if not mask.any(dim=-1).all():
        raise ValueError("Every message row needs a neighbor or self-loop.")
    return mask


class GraphAttentionLayer(nn.Module):
    """Dense multi-head additive GAT; positive graph entries permit messages.

    Head outputs are concatenated. Attention normalization is over neighbors,
    not all cells. No dropout: deterministic inference/target estimates.
    Dense attention costs O(B*heads*N^2), appropriate for 46 nodes, not unbounded.
    """

    def __init__(self, input_dim, output_dim, heads=4, negative_slope=0.2):
        super().__init__()
        if input_dim < 1 or output_dim < 1 or heads < 1 or output_dim % heads:
            raise ValueError("Output dimension must divide into positive heads.")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.output_dim = output_dim
        self.negative_slope = negative_slope
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attention_target = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(self, x, graph):
        if (
            x.ndim != 3
            or x.shape[-1] != self.projection.in_features
            or x.shape[0] < 1
            or x.shape[1] < 1
        ):
            raise ValueError(
                "Features must be [B,N,input_dim] with nonempty batch/nodes."
            )
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Features must be finite floating point.")
        b, n, _ = x.shape
        mask = graph_mask(graph, b, n, x.device)
        # Project [B,N,F] to [B,heads,N,head_dim].
        h = self.projection(x).view(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        source = (h * self.attention_source[None, :, None, :]).sum(-1)
        target = (h * self.attention_target[None, :, None, :]).sum(-1)
        logits = F.leaky_relu(
            source.unsqueeze(-1) + target.unsqueeze(-2),
            negative_slope=self.negative_slope,
        )
        logits = logits.masked_fill(~mask[:, None, :, :], -torch.inf)
        attention = logits.softmax(dim=-1)
        messages = torch.matmul(attention, h)
        merged = messages.permute(0, 2, 1, 3).reshape(b, n, self.output_dim)
        return F.elu(merged + self.bias)


class GraphDuelingQNetwork(nn.Module):
    """One Q value per cell; no fixed 46-output fully-connected classifier.

    Global value uses mean pooling (permutation invariant). Shared node advantage
    head is permutation equivariant. Dueling centering uses all N node advantages;
    action masking is separate and must be applied at selection AND DDQN targets.
    """

    def __init__(self, settings=None):
        super().__init__()
        self.settings = NetworkSettings() if settings is None else settings
        f = self.settings
        for v in (f.input_features, f.hidden_dim, f.attention_heads, f.head_hidden_dim):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError("Network dimensions must be positive integers.")
        if f.hidden_dim % f.attention_heads or not 0 < f.negative_slope < 1:
            raise ValueError("Invalid head count or negative slope.")
        self.gat1 = GraphAttentionLayer(
            f.input_features, f.hidden_dim, f.attention_heads, f.negative_slope
        )
        self.gat2 = GraphAttentionLayer(
            f.hidden_dim, f.hidden_dim, f.attention_heads, f.negative_slope
        )
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

    def forward(self, x, graph):
        h1 = self.gat1(x, graph)
        h2 = self.norm(h1 + self.gat2(h1, graph))
        value = self.value_head(h2.mean(dim=1))
        advantage = self.advantage_head(h2).squeeze(-1)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def masked_q_values(q_values, action_mask):
    """Return a new Q tensor with illegal actions -inf. Never mutate raw Qs.

    Terminal transitions with no eligible next action must be excluded from
    target maximization by the DDQN agent, not passed here with an empty mask.
    """
    if (
        q_values.ndim != 2
        or not q_values.is_floating_point()
        or not torch.isfinite(q_values).all()
    ):
        raise ValueError("Q values must be finite [B,N].")
    if (
        not isinstance(action_mask, torch.Tensor)
        or action_mask.dtype != torch.bool
        or action_mask.device != q_values.device
    ):
        raise ValueError("Action mask must be a bool tensor on the Q device.")
    mask = action_mask
    if mask.ndim == 1 and mask.shape[0] == q_values.shape[1]:
        mask = mask.unsqueeze(0).expand(q_values.shape[0], -1)
    if mask.shape != q_values.shape or not mask.any(dim=-1).all():
        raise ValueError("Every Q row must have a matching nonempty action mask.")
    return q_values.masked_fill(~mask, -torch.inf)


def self_test(device="cpu"):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    dev = torch.device(device)
    net = GraphDuelingQNetwork().to(dev)
    for n in (3, 10, 46):
        x = torch.rand(2, n, 15, device=dev)
        graph = torch.eye(n, device=dev)
        q = net(x, graph)
        assert q.shape == (2, n) and torch.isfinite(q).all()
        mask = torch.zeros(2, n, dtype=torch.bool, device=dev)
        mask[:, 0] = True
        assert (masked_q_values(q, mask).argmax(-1) == 0).all()
    n = 46
    x = torch.rand(2, n, 15, device=dev)
    graph = (torch.rand(n, n, device=dev) > 0.85).float()
    graph = torch.maximum(graph, graph.T)
    graph.fill_diagonal_(1.0)
    net.eval()
    q = net(x, graph)
    permutation = torch.randperm(n, device=dev)
    permuted_q = net(x[:, permutation, :], graph[permutation][:, permutation])
    assert torch.allclose(permuted_q, q[:, permutation], atol=2e-5, rtol=2e-5)
    # A batch of graph matrices must agree with a shared graph.
    assert torch.allclose(net(x, graph.expand(2, -1, -1)), q, atol=1e-6, rtol=1e-6)
    # Identity graph: no communication from other nodes in this GAT layer.
    layer = net.gat1
    identity = torch.eye(n, device=dev)
    before = layer(x, identity)
    changed = x.clone()
    changed[:, 1, :] += 1.0
    after = layer(changed, identity)
    assert torch.allclose(before[:, 0, :], after[:, 0, :])
    net.train()
    net.zero_grad(set_to_none=True)
    loss = net(x, graph).square().mean()
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    )
    assert net.gat1.projection.weight.grad.abs().sum() > 0
    raw = torch.tensor([[2.0, 1e6, 3.0]], device=dev)
    allowed = torch.tensor([[True, False, True]], device=dev)
    assert (
        masked_q_values(raw, allowed).argmax(-1).item() == 2 and raw[0, 1].item() == 1e6
    )
    for callback in (
        lambda: net(x, torch.zeros(n, n, device=dev)),
        lambda: net(torch.zeros(2, 0, 15, device=dev), torch.zeros(0, 0, device=dev)),
        lambda: masked_q_values(raw, torch.zeros_like(allowed)),
    ):
        try:
            callback()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted.")
    print(
        "Network tests passed: variable node counts, batched graphs, finite gradients, permutation, neighbor isolation, action masks and invalid inputs."
    )
