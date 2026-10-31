"""Evaluation / transfer.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

import numpy as np
import torch


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class TransferPolicy:
    def __init__(self, native, message_graph, feature_names):
        from toponet.envs.features import FEATURE_NAMES

        require(list(feature_names) == list(FEATURE_NAMES), "Feature order mismatch")
        require(
            native.memory is None and not native.policy.training,
            "Require a loaded frozen eval policy with no replay",
        )
        self.native = native
        self.feature_names = tuple(feature_names)
        self.expected_graph = np.array(message_graph, dtype=np.float32, copy=True)
        g = self.expected_graph
        require(
            g.ndim == 2
            and g.shape[0] == g.shape[1]
            and 1 <= len(g) <= 96
            and np.isfinite(g).all()
            and (g >= 0).all()
            and np.array_equal(g, g.T)
            and (np.diag(g) > 0).all(),
            "Unsupported transfer message graph",
        )
        self.transfer_graph_tensor = torch.tensor(g, device=native.device)
        self.saved_original_graph = native.graph.copy()

    def __getattr__(self, name):
        if name in [
            "policy",
            "target",
            "optimizer",
            "action_rng",
            "memory",
            "environment_steps",
            "optimizer_updates",
            "target_syncs",
            "device",
            "seed",
        ]:
            return getattr(self.native, name)
        raise AttributeError(name)

    def select_action(self, state, eval_mode=False):
        from toponet.models.graph_q import masked_q_values

        require(eval_mode is True, "Transfer adapter is evaluation only")
        x = np.asarray(state["node_features"])
        m = np.asarray(state["action_mask"])
        g = np.asarray(state["message_graph"])
        n = len(self.expected_graph)
        require(
            x.shape == (n, len(self.feature_names))
            and x.dtype == np.float32
            and np.isfinite(x).all()
            and (x >= 0).all()
            and (x <= 1).all(),
            "Invalid transfer feature schema/range",
        )
        require(
            m.shape == (n,) and m.dtype == np.bool_ and m.any(),
            "Invalid legal action mask",
        )
        require(
            np.array_equal(g, self.expected_graph)
            and np.array_equal(self.native.graph, self.saved_original_graph),
            "State graph changed or original graph was edited",
        )
        require(not self.native.policy.training, "Frozen policy entered training mode")
        with torch.no_grad():
            q = self.native.policy(
                torch.tensor(x, device=self.native.device).unsqueeze(0),
                self.transfer_graph_tensor,
            )
            return int(
                masked_q_values(q, torch.tensor(m, device=self.native.device))
                .argmax(1)
                .item()
            )
