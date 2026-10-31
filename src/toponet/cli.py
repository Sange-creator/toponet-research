"""Explicit entry points: importing TopoNet never launches a research run."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--project",
        type=Path,
        default=default if (default / "configs").is_dir() else Path.cwd(),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Read progress and evidence status")
    sub.add_parser("verify-files", help="Check curated source/data/model identities")
    p = sub.add_parser(
        "analyze", help="Recompute descriptive statistics from saved case summaries"
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p = sub.add_parser(
        "replay", help="Run one requested frozen case into a new output directory"
    )
    p.add_argument("--area", choices=["heldout", "pokhara"], default="heldout")
    p.add_argument("--case", type=int, default=1)
    p.add_argument("--controllers", nargs="+")
    p.add_argument("--output", type=Path, required=True)
    for name in ("train", "smoke"):
        p = sub.add_parser(
            name,
            help="Start an explicit new development run"
            if name == "train"
            else "80-step pipeline check",
        )
        p.add_argument("--config", type=Path)
        p.add_argument("--output", type=Path, required=True)
        if name == "train":
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    if args.command == "status":
        result = json.loads((project / "tracking/progress.json").read_text())
    elif args.command == "verify-files":
        from toponet.io import sha256

        records = json.loads((project / "provenance/PROJECT_SHA256.json").read_text())
        bad = [
            name
            for name, digest in records.items()
            if not (project / name).is_file() or sha256(project / name) != digest
        ]
        if bad:
            raise RuntimeError("Curated files changed: " + str(bad))
        result = dict(checked_files=len(records), failures=bad)
    elif args.command == "analyze":
        from toponet.evaluation.analysis import analyze

        result = analyze(project, args.bootstrap)
    elif args.command == "replay":
        from toponet.evaluation.replay import replay

        result = replay(project, args.output, args.area, args.case, args.controllers)
    else:
        from toponet.training.runner import train

        config = (
            args.config or project / "configs/training/reference_ho0875_seed17.json"
        )
        result = train(
            project,
            config,
            args.output,
            getattr(args, "resume", False),
            args.command == "smoke",
        )
    print(json.dumps(result, indent=2))
