"""Initialize independent run records; no training takes place here."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hotel_ml.experiments.config import load_config


def git_metadata() -> dict:
    """Record this checkout's identity without assuming git is installed."""
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def environment_metadata() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            )
        ),
    }


def initialize_run(
    config_path: str | Path,
    output_root: str | Path,
    data_path: str | Path | None = None,
) -> Path:
    """Create a unique run directory with config and provenance, before training."""
    config = load_config(config_path)
    dataset = {"status": "not_provided"}
    if data_path is not None:
        source = Path(data_path).resolve()
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        dataset = {
            "status": "fingerprinted",
            "filename": source.name,
            "sha256": digest,
            "bytes": source.stat().st_size,
        }
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:12]}"
    run = Path(output_root) / config["task"] / config["version"] / run_id
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now.isoformat(),
        "status": "initialized",
        "config": config,
        "git": git_metadata(),
        "environment": environment_metadata(),
        "dataset": dataset,
    }
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return run
