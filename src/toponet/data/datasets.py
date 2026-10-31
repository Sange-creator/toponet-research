"""Portable readers for the frozen original-area and Pokhara simulation inputs."""

import copy
import csv
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from toponet.data.geometry import latlon_to_xy
from toponet.data.graph import build_neighbor_graph
from toponet.data.movement import RouteMotion
from toponet.envs.features import LearningInterface, RewardSettings
from toponet.envs.handover import HandoverEnvironment, HandoverSettings
from toponet.envs.radio import RadioSettings
from toponet.training.episode import file_hash


def read_json(path):
    return json.loads(Path(path).read_text())


class OriginalDataset:
    def __init__(self, project, reference_config=None, split_config=None):
        self.project = Path(project).resolve()
        self.reference_path = self.project / (
            reference_config or "configs/reference.json"
        )
        self.split_path = self.project / (split_config or "configs/splits.json")
        self.config = read_json(self.reference_path)
        self.split = read_json(self.split_path)
        self.bank = self.project / "data/original/routes"
        manifest = (
            self.bank / "candidate_routes/route_manifest_training_candidates.json"
        )
        if file_hash(manifest) != self.split["source_manifest_sha256"]:
            raise ValueError("Candidate route manifest differs from frozen split")
        self.rows = {r["route_id"]: r for r in read_json(manifest)["routes"]}
        for left, right in [
            ("train_route_ids", "validation_pool_ids"),
            ("train_route_ids", "reserved_test_route_ids"),
            ("validation_pool_ids", "reserved_test_route_ids"),
        ]:
            if set(self.split[left]) & set(self.split[right]):
                raise ValueError("Overlapping route partitions")
        cfg = self.config
        self.xy = latlon_to_xy(
            [c["lat"] for c in cfg["cell_records"]],
            [c["lon"] for c in cfg["cell_records"]],
            cfg["coordinate_reference"],
        )
        self.neighbors = np.load(
            self.project / "data/original/inputs/handover_neighbors.npy",
            allow_pickle=False,
        )
        if not np.array_equal(
            self.neighbors,
            build_neighbor_graph(self.xy, cfg["neighbor_graph_summary"]["radius_m"])[0],
        ):
            raise ValueError("Frozen neighbor graph mismatch")
        self.hashes = read_json(self.bank / "SHA256.json")

    @lru_cache(maxsize=128)
    def geometry(self, route):
        path = self.bank / "candidate_routes" / self.rows[route]["file"]
        if (
            file_hash(path)
            != self.hashes["candidate_routes/" + self.rows[route]["file"]]
        ):
            raise ValueError("Route geometry hash mismatch: " + route)
        with np.load(path, allow_pickle=False) as data:
            return data["xy_m"].copy(), data["cumulative_distance_m"].copy()

    def interface(self, route, handover_cost=None):
        cfg = self.config
        points, cumulative = self.geometry(route)
        motion = RouteMotion(
            points.copy(), cumulative.copy(), cfg["speed_mps"], cfg["dt_s"]
        )
        if not np.isclose(
            motion.route_length_m, self.rows[route]["length_m"], rtol=0, atol=1e-6
        ):
            raise ValueError("Frozen route length mismatch")
        env = HandoverEnvironment(
            copy.deepcopy(cfg["cell_records"]),
            self.xy.copy(),
            self.neighbors.copy(),
            motion,
            HandoverSettings(**cfg["handover_settings"]),
            RadioSettings(**cfg["radio_settings"]),
        )
        if asdict(env.traffic.settings) != cfg["traffic_settings"]:
            raise ValueError("Traffic semantics differ from reference")
        vi = LearningInterface(
            env, RewardSettings(**cfg["interface_metadata"]["reward_settings"])
        )
        if vi.metadata() != cfg["interface_metadata"]:
            raise ValueError("Feature/reward schema differs from reference")
        if handover_cost is not None:
            reward = dict(
                cfg["interface_metadata"]["reward_settings"],
                handover_cost=handover_cost,
            )
            vi.reward_settings = RewardSettings(**reward)
        return vi


def pokhara_interface(project, scenario):
    root = Path(project) / "data/pokhara"
    physical = read_json(root / "PHYSICAL_CONFIGURATION.json")
    folder = root / "layouts" / ("layout_" + str(scenario["layout_seed"]))
    cells = read_json(folder / "cells.json")
    with np.load(folder / "layout.npz", allow_pickle=False) as data:
        xy, neighbors, messages = [
            data[k].copy() for k in ("cell_xy_m", "handover_neighbors", "message_graph")
        ]
    expected, graph = build_neighbor_graph(xy, physical["neighbor_radius_m"])
    if not np.array_equal(neighbors, expected) or not np.array_equal(messages, graph):
        raise ValueError("Pokhara graph mismatch")
    expected_xy = latlon_to_xy(
        [c["lat"] for c in cells],
        [c["lon"] for c in cells],
        physical["coordinate_reference"],
    )
    if not np.allclose(xy, expected_xy, rtol=0, atol=1e-6):
        raise ValueError("Pokhara coordinate mismatch")
    rows = read_json(root / "routes/route_manifest.json")["routes"]
    row = next(r for r in rows if r["route_id"] == scenario["route"])
    with np.load(root / "routes" / row["file"], allow_pickle=False) as data:
        motion = RouteMotion(
            data["xy_m"].copy(),
            data["cumulative_distance_m"].copy(),
            physical["speed_mps"],
            physical["dt_s"],
        )
    if abs(motion.route_length_m - row["length_m"]) > 1e-6:
        raise ValueError("Pokhara route length mismatch")
    env = HandoverEnvironment(
        cells,
        xy,
        neighbors,
        motion,
        HandoverSettings(**physical["handover_settings"]),
        RadioSettings(**physical["radio_settings"]),
    )
    if asdict(env.traffic.settings) != physical["traffic_settings"]:
        raise ValueError("Pokhara traffic mismatch")
    vi = LearningInterface(
        env, RewardSettings(**physical["interface_metadata"]["reward_settings"])
    )
    if vi.metadata() != physical["interface_metadata"]:
        raise ValueError("Pokhara feature schema mismatch")
    return vi
