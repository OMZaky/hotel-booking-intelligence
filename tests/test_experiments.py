"""Behavior checks for reproducibility and safe experiment initialization."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hotel_ml.experiments.config import load_config
from hotel_ml.experiments.runs import initialize_run


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "task": "classification",
            "version": "v1",
            "model": "logistic_regression",
            "stage": "baseline",
            "representation": "original",
            "seed": 42,
            "split_id": "pending-data-audit",
            "prediction_point": "pending-feature-availability-review",
            "selection_metric": "f1",
            "parameters": {},
        }
        self.path = self.root / "recipe.json"

    def write_config(self, config=None):
        self.path.write_text(json.dumps(config or self.config), encoding="utf-8")

    def test_invalid_task_and_unsafe_version_are_rejected(self):
        for field, value in [("task", "forecasting"), ("version", "../escape")]:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[field] = value
                self.write_config(config)
                with self.assertRaises(ValueError):
                    load_config(self.path)

    def test_target_cannot_be_declared_as_clustering_feature(self):
        config = dict(
            self.config, task="clustering", selection_metric="silhouette", features=["IsCanceled"]
        )
        self.write_config(config)
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_missing_and_unknown_fields_are_rejected(self):
        for operation in ["missing", "unknown"]:
            with self.subTest(operation=operation):
                config = copy.deepcopy(self.config)
                if operation == "missing":
                    del config["seed"]
                else:
                    config["sead"] = 42
                self.write_config(config)
                with self.assertRaises(ValueError):
                    load_config(self.path)

    def test_boolean_seed_and_infinite_parameter_are_rejected(self):
        for update in [{"seed": True}, {"parameters": {"alpha": float("inf")}}]:
            with self.subTest(update=update):
                self.write_config(dict(self.config, **update))
                with self.assertRaises(ValueError):
                    load_config(self.path)

    def test_runs_are_unique_and_record_provenance_without_claiming_training(self):
        self.write_config()
        output = self.root / "artifacts"
        first = initialize_run(self.path, output)
        second = initialize_run(self.path, output)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, output / "classification" / "v1")
        manifest = json.loads((first / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "initialized")
        self.assertEqual(manifest["config"], self.config)
        self.assertIn("python", manifest["environment"])
        self.assertIn("packages", manifest["environment"])
        self.assertIn("commit", manifest["git"])
        self.assertEqual(manifest["dataset"]["status"], "not_provided")
        self.assertTrue((first / "config.json").is_file())

    def test_dataset_hash_reflects_bytes(self):
        self.write_config()
        dataset = self.root / "H2.csv"
        content = b"ADR,IsCanceled\n100,0\n"
        dataset.write_bytes(content)
        run = initialize_run(self.path, self.root / "artifacts", dataset)
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["dataset"]["sha256"], hashlib.sha256(content).hexdigest())

    def test_missing_dataset_does_not_create_run(self):
        self.write_config()
        output = self.root / "artifacts"
        with self.assertRaises(FileNotFoundError):
            initialize_run(self.path, output, self.root / "missing.csv")
        self.assertFalse(output.exists())

    def test_final_status_fields_and_targets_are_rejected_as_predictors(self):
        for task, feature in [
            ("regression", "ADR"),
            ("classification", "IsCanceled"),
            ("classification", "ReservationStatusDate"),
            ("clustering", "ReservationStatus"),
        ]:
            with self.subTest(task=task, feature=feature):
                metric = {"regression": "mae", "classification": "f1", "clustering": "silhouette"}[
                    task
                ]
                self.write_config(
                    dict(self.config, task=task, selection_metric=metric, features=[feature])
                )
                with self.assertRaises(ValueError):
                    load_config(self.path)


if __name__ == "__main__":
    unittest.main()
