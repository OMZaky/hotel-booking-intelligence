"""Validate experiment recipes without importing scientific dependencies."""

import json
import re
from pathlib import Path

TASKS = {"regression", "classification", "clustering"}
REQUIRED = {
    "task",
    "version",
    "model",
    "stage",
    "representation",
    "seed",
    "split_id",
    "prediction_point",
    "selection_metric",
    "parameters",
}
OPTIONAL = {"features", "notes"}
METRICS = {
    "regression": {"mae", "rmse", "r2"},
    "classification": {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"},
    "clustering": {"silhouette", "davies_bouldin"},
}


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON value is not permitted: {value}")


def load_config(path: str | Path) -> dict:
    """Load a complete JSON recipe; reject typos and known direct leakage."""
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    if not isinstance(config, dict):
        raise ValueError("An experiment config must be a JSON object")
    missing = REQUIRED - config.keys()
    unknown = config.keys() - REQUIRED - OPTIONAL
    if missing or unknown:
        raise ValueError(
            f"Invalid config keys: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if not isinstance(config["task"], str) or config["task"] not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    if not isinstance(config["version"], str) or not re.fullmatch(
        r"v[1-9][0-9]*", config["version"]
    ):
        raise ValueError("version must be v1, v2, ... (no directory components)")
    if not isinstance(config["model"], str) or not re.fullmatch(
        r"[a-z][a-z0-9_]*", config["model"]
    ):
        raise ValueError("model must be a lowercase identifier")
    if config["stage"] not in ("baseline", "tuned"):
        raise ValueError("stage must be baseline or tuned")
    if config["representation"] not in ("original", "pca", "lda"):
        raise ValueError("representation must be original, pca, or lda")
    if config["representation"] == "lda" and config["task"] != "classification":
        raise ValueError("LDA representation is only supported for classification")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    for field in ("split_id", "prediction_point", "selection_metric"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if config["selection_metric"] not in METRICS[config["task"]]:
        raise ValueError("selection_metric is not valid for the selected task")
    if not isinstance(config["parameters"], dict):
        raise ValueError("parameters must be an object")
    features = config.get("features", [])
    if not isinstance(features, list) or any(
        not isinstance(feature, str) or not feature.strip() for feature in features
    ):
        raise ValueError("features must be a list of nonempty source column names")
    if len(features) != len(set(features)):
        raise ValueError("features must be unique")
    forbidden = {"ReservationStatus", "ReservationStatusDate", "IsCanceled"}
    if config["task"] in ("regression", "clustering"):
        forbidden.add("ADR")
    leaked = forbidden.intersection(features)
    if leaked:
        raise ValueError(f"Outcome/final-status columns cannot be predictors: {sorted(leaked)}")
    if "notes" in config and not isinstance(config["notes"], str):
        raise ValueError("notes must be a string")
    return config
