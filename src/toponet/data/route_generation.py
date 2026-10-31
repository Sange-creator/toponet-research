"""Data / route generation.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from pathlib import Path
import csv
import heapq
import json
import math
import numpy as np

PILOTS = (
    ("southeast", (1, 2, 3, 4)),
    ("west_north_east", (12, 18, 20, 23, 27)),
    ("centre_north", (32, 34, 41, 42, 23)),
)


def weighted_adjacency(graph):
    xy = graph["xy_m"]
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Invalid road positions.")
    n = len(xy)
    a, b, length = graph["edge_from"], graph["edge_to"], graph["edge_length_m"]
    if not (len(a) == len(b) == len(length)):
        raise ValueError("Inconsistent edge arrays.")
    if (a < 0).any() or (b < 0).any() or (a >= n).any() or (b >= n).any():
        raise ValueError("Invalid edge endpoints.")
    if not np.isfinite(length).all() or (length <= 0).any():
        raise ValueError("Edge lengths must be finite and positive.")
    geometric = np.linalg.norm(xy[a] - xy[b], axis=1)
    if not np.allclose(length, geometric, atol=1e-6, rtol=1e-8):
        raise ValueError("Edge distances do not match road coordinates.")
    adjacency = [[] for _ in range(n)]
    for u, v, d in zip(a, b, length):
        adjacency[int(u)].append((int(v), float(d)))
    return adjacency


def shortest_path(adjacency, start, goal):
    """Dijkstra by distance along existing directed edges, not straight-line distance."""
    n = len(adjacency)
    if not 0 <= start < n or not 0 <= goal < n:
        raise ValueError("Endpoint index outside graph.")
    distances = np.full(n, np.inf)
    parents = np.full(n, -1, dtype=np.int64)
    distances[start] = 0.0
    pending = [(0.0, start)]
    while pending:
        distance, node = heapq.heappop(pending)
        if distance > distances[node]:
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(int(parents[path[-1]]))
            return list(reversed(path)), float(distance)
        for other, cost in adjacency[node]:
            candidate = distance + cost
            if candidate < distances[other]:
                distances[other] = candidate
                parents[other] = node
                heapq.heappush(pending, (candidate, other))
    raise ValueError(f"No directed route from road node {start} to {goal}.")


def position_at_distance(xy_m, cumulative_m, distance_m):
    """Interpolate along the polyline. Reused later for user movement.

    Does not jump between waypoints or create straight-line shortcuts.
    """
    xy = np.asarray(xy_m, dtype=float)
    cumulative = np.asarray(cumulative_m, dtype=float)
    d = float(distance_m)
    if (
        xy.ndim != 2
        or xy.shape[1] != 2
        or len(xy) < 2
        or cumulative.shape != (len(xy),)
    ):
        raise ValueError("Provide a valid route polyline and cumulative distances.")
    if (
        not np.isfinite(xy).all()
        or not np.isfinite(cumulative).all()
        or not math.isfinite(d)
    ):
        raise ValueError("Route inputs must be finite.")
    if abs(cumulative[0]) > 1e-8 or (np.diff(cumulative) <= 0).any():
        raise ValueError(
            "Cumulative distances must start at zero and strictly increase."
        )
    expected = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if not np.allclose(np.diff(cumulative), expected, atol=1e-6, rtol=1e-8):
        raise ValueError("Cumulative distances do not match route geometry.")
    d = float(np.clip(d, 0.0, cumulative[-1]))
    segment = min(int(np.searchsorted(cumulative, d, side="right")) - 1, len(xy) - 2)
    fraction = (d - cumulative[segment]) / (
        cumulative[segment + 1] - cumulative[segment]
    )
    return xy[segment] + fraction * (xy[segment + 1] - xy[segment])
