# Folders to add when needed

Keep the current repository small. Create a folder when its first implementation or real output
needs it; do not recreate empty placeholders in advance.

The shared Python package, versioned experiment configs, and independent run IDs remain the foundation.

| Folder | Add when | Responsibility |
| --- | --- | --- |
| `src/hotel_ml/data/` | Data audit and splitting | Load H2, validate its source schema, and persist reproducible split manifests. |
| `src/hotel_ml/features/` | Feature policies and engineering | Approve task-specific inputs and build justified features without leakage. |
| `src/hotel_ml/preprocessing/` | Reusable preprocessing | Compose imputation, encoding, scaling, and fitted transformations within training folds. |
| `src/hotel_ml/tasks/regression/` | First ADR model | Regression model definitions, training, and bounded hyperparameter searches. |
| `src/hotel_ml/tasks/classification/` | First cancellation model | Classification models, training, tuning, and validation-based threshold choices. |
| `src/hotel_ml/tasks/clustering/` | First segmentation experiment | Unsupervised algorithms, parameter searches, and cluster interpretation. |
| `src/hotel_ml/evaluation/` | First model comparisons | Required metrics, fair comparisons, and the sealed final-test procedure. |
| `src/hotel_ml/inference/` | Saved-pipeline predictions | Load complete fitted pipelines and validate new booking inputs without retraining. |
| `apps/` | A graphical interface is agreed | UI code calling the shared package. Keep CLI behavior in `src/hotel_ml/cli.py`. |
| `data/interim/` | Intermediate data is generated | Temporary transformations and split manifests; contents remain ignored by Git. |
| `data/processed/` | Derived model inputs are generated | Reproducible derived datasets; contents remain ignored by Git. |
| `reports/figures/` | First reviewed plots | Publishable EDA, evaluation, and clustering figures. |
| `reports/tables/` | First reviewed comparisons | Baseline/tuned summaries with task, recipe version, and run references. |
| `reports/milestone1/` | Progress report is prepared | Report due November 5, 2026, including preliminary audit and contributions. |
| `reports/final/` | Final submission is prepared | Report due December 11, 2026, and discussion materials. |

These boundaries are a starting point. Introduce only the folders justified by the actual code;
small related functions can initially share a module.

## Conventions to preserve

- Add package `__init__.py` files when introducing real Python subpackages.
- Version recipes under `configs/<task>/<version>/`; share implementation rather than copying
  complete source trees for each version.
- Keep `data/raw/H2.csv` immutable. Do not download or publish data without instruction.
- Define prediction points and feature availability before training. Fit learned preprocessing
  within training folds; keep final test data out of model selection.
- Keep raw run output and fitted pipelines in ignored `artifacts/`.
  Promote reviewed summaries to `reports/` when those outputs exist.
- The final application must expose analysis, comparisons, training/tuning access, clustering,
  and regression/classification inference from saved pipelines.
- Retain fitted pipelines locally for the live demonstration and exclude them from submission ZIPs.

See [roadmap](roadmap.md) for delivery phases and [architecture](architecture.md) for current code.
