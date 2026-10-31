"""Data / input inspection.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from pathlib import Path
from dataclasses import asdict
import csv
import json
import numpy as np
from toponet.data.towers import load_cells
from toponet.data.geometry import latlon_to_xy
from toponet.data.graph import build_neighbor_graph
from toponet.data.movement import RouteMotion
from toponet.envs.handover import HandoverEnvironment
from toponet.baselines.controllers import (
    ImmediateSignalController,
    A3StyleController,
    A3Settings,
    LoadAwareController,
    LoadAwareSettings,
)


def load_environment_inputs(
    filename, layout_dir, neighbor_dir, route_dir, route_id, speed_mps, dt_s
):
    cells = load_cells(filename)
    layout = Path(layout_dir)
    ng = Path(neighbor_dir)
    routes = Path(route_dir)
    ref = json.loads((layout / "coordinate_reference.json").read_text())
    manifest = json.loads((routes / "route_manifest.json").read_text())
    if ref != manifest["coordinate_reference"]:
        raise ValueError("Coordinate reference mismatch.")
    with (layout / "cell_positions.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if [int(r["node_index"]) for r in rows] != list(range(len(cells))) or [
        int(r["site_id"]) for r in rows
    ] != [c["site_id"] for c in cells]:
        raise ValueError("Cell order mismatch.")
    xy = latlon_to_xy([c["lat"] for c in cells], [c["lon"] for c in cells], ref)
    saved = np.array([[float(r["x_east_m"]), float(r["y_north_m"])] for r in rows])
    if not np.allclose(xy, saved, atol=1e-5, rtol=0):
        raise ValueError("Saved coordinates mismatch.")
    graph_meta = json.loads((ng / "graph_summary.json").read_text())
    neighbors = np.load(ng / "handover_neighbors.npy", allow_pickle=False)
    if not np.array_equal(
        neighbors, build_neighbor_graph(xy, graph_meta["radius_m"])[0]
    ):
        raise ValueError("Saved neighbors mismatch.")
    route = next((r for r in manifest["routes"] if r["route_id"] == route_id), None)
    if route is None:
        raise ValueError("Unknown route.")
    motion = RouteMotion.from_file(routes / route["file"], speed_mps, dt_s)
    return HandoverEnvironment(cells, xy, neighbors, motion), manifest, graph_meta
