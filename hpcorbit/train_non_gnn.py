#!/usr/bin/env python3
"""Run a standard TopoNet DDQN training schedule with graph message passing removed."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    implementation_dir = Path(__file__).resolve().parent
    settings = json.loads(args.config.read_text())
    ablation = settings.get("ablation", {})
    expected = {
        "architecture": "nodewise_mlp_no_graph",
        "model_sha256": sha256(implementation_dir / "nodewise_mlp.py"),
        "launcher_sha256": sha256(Path(__file__).resolve()),
    }
    matched_hash = ablation.get("matched_gat_config_sha256", "")
    if (
        any(ablation.get(key) != value for key, value in expected.items())
        or not isinstance(ablation.get("matched_gat_config"), str)
        or len(matched_hash) != 64
        or any(char not in "0123456789abcdef" for char in matched_hash)
    ):
        raise RuntimeError(
            "Ablation configuration does not match the current implementation. "
            "Regenerate it from the original GAT config and use a new run directory."
        )

    from nodewise_mlp import install_nodewise_mlp

    install_nodewise_mlp()
    from toponet.training.runner import train

    result = train(
        args.project.resolve(),
        args.config.resolve(),
        args.output.resolve(),
        resume=args.resume,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
