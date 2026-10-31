"""Data / graph.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from pathlib import Path
import csv
import json
import math
import numpy as np


def build_neighbor_graph(xy_m, radius_m=1500.0):
    """Return neighbor-only and message matrices in the supplied node order."""
    xy = np.asarray(xy_m, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
        raise ValueError("Provide finite positions with shape [number of cells, 2].")
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Radius must be positive and finite.")
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    neighbors = distances <= radius_m
    np.fill_diagonal(neighbors, False)
    messages = neighbors.copy()
    np.fill_diagonal(messages, True)
    return neighbors, messages


def graph_components(neighbors):
    """Count physical-candidate connectivity without treating self-loops as links."""
    adjacency = np.asarray(neighbors, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Provide a square neighbor matrix.")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("This provisional graph must be undirected.")
    unseen = set(range(len(adjacency)))
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        group = []
        while stack:
            node = stack.pop()
            group.append(node)
            for other in np.flatnonzero(adjacency[node]):
                other = int(other)
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        components.append(sorted(group))
    return components
