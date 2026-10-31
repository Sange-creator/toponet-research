#!/usr/bin/env python3
"""Create graph-free MLP configs paired with existing GAT training configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_one(source: Path, output_dir: Path) -> dict:
    source = source.resolve()
    settings = json.loads(source.read_text())
    required = {
        "episodes",
        "agent_seed",
        "route_seed",
        "handover_cost",
        "validation_target_steps",
    }
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"{source.name} is missing required fields: {missing}")

    overrides = dict(settings.get("network_overrides", {}))
    if "architecture" in overrides:
        raise ValueError(
            f"{source.name} contains a network architecture override. "
            "Start from the original GAT training configuration."
        )
    settings["network_overrides"] = overrides
    implementation_dir = Path(__file__).resolve().parent
    matched_config_sha256 = sha256(source)
    settings["ablation"] = {
        "architecture": "nodewise_mlp_no_graph",
        "model_sha256": sha256(implementation_dir / "nodewise_mlp.py"),
        "launcher_sha256": sha256(implementation_dir / "train_non_gnn.py"),
        "matched_gat_config": source.name,
        "matched_gat_config_sha256": matched_config_sha256,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{source.stem}_non_gnn.json"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite {destination}. Move it or choose a new output directory."
        )
    destination.write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    return {
        "config": destination.name,
        "config_sha256": sha256(destination),
        "source_config_sha256": matched_config_sha256,
        "agent_seed": settings["agent_seed"],
        "route_seed": settings["route_seed"],
        "episodes": settings["episodes"],
        "architecture": settings["ablation"]["architecture"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline_configs",
        nargs="+",
        type=Path,
        help="Exact GAT JSON configs used for the matched policies, one per seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/ablation_non_gnn"),
    )
    args = parser.parse_args()

    records = [prepare_one(path, args.output_dir) for path in args.baseline_configs]
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
