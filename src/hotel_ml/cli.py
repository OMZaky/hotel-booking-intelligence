"""Bootstrap CLI. Model training is scheduled in the project roadmap."""

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

from hotel_ml import __version__
from hotel_ml.experiments.config import load_config
from hotel_ml.experiments.runs import git_metadata, initialize_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hotel Booking Intelligence")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect runtime and expected dataset location")
    doctor.add_argument("--data", type=Path, default=None)
    validate = commands.add_parser("validate-config", help="Validate an experiment recipe")
    validate.add_argument("--config", type=Path, required=True)
    initialize = commands.add_parser("init-run", help="Create a provenance record; does not train")
    initialize.add_argument("--config", type=Path, required=True)
    initialize.add_argument("--data", type=Path)
    initialize.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("HOTEL_ML_ARTIFACT_ROOT", "artifacts")),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            data = args.data or Path(os.environ.get("HOTEL_ML_DATA_PATH", "data/raw/H2.csv"))
            result = {
                "version": __version__,
                "python": platform.python_version(),
                "supported_python": (3, 12) <= sys.version_info[:2] < (3, 15),
                "preferred_python": "3.12",
                "dataset": str(data.resolve()),
                "dataset_exists": data.is_file(),
                "ml_dependencies": {
                    name: importlib.util.find_spec(name) is not None
                    for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib")
                },
                "git": git_metadata(),
            }
            print(json.dumps(result, indent=2))
        elif args.command == "validate-config":
            print(json.dumps(load_config(args.config), indent=2))
        else:
            print(initialize_run(args.config, args.output_root, args.data).resolve())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
