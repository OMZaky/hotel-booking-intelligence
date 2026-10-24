<div align="center">

# Hotel Booking Intelligence

**Understanding hotel bookings through pricing, cancellation prediction, and segmentation.**

A CSE382 Introduction to Machine Learning project built around the City Hotel (H2) dataset.

[![Checks](https://github.com/OMZaky/hotel-booking-intelligence/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/OMZaky/hotel-booking-intelligence/actions/workflows/ci.yml) ![Python 3.12 to 3.14](https://img.shields.io/badge/Python-3.12%E2%80%933.14-3776AB?logo=python&logoColor=white) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OMZaky/hotel-booking-intelligence/blob/main/notebooks/colab_bootstrap.ipynb)

[Overview](#overview) · [Quick start](#quick-start) · [Google Colab](#google-colab) · [Experiments](#experiments) · [Project structure](#project-structure) · [Milestones](#milestones)

</div>

---

## Overview

This project investigates three complementary questions using real hotel booking records:

| Task | Question | Target / output |
| --- | --- | --- |
| **Regression** | What average daily rate is associated with a booking? | `ADR` |
| **Classification** | Will a booking eventually be canceled? | `IsCanceled` |
| **Clustering** | Which natural booking segments can we identify? | Interpretable groups discovered without outcome labels |

The finished system will compare **at least three models or algorithms per task**, with baseline
and tuned results for each. It will include PCA/LDA comparisons, a documented leakage audit,
and an application that loads saved pipelines for new predictions.

> **Current stage: project foundation.** Environment inspection, experiment-config validation,
> and run-record initialization are implemented. Dataset analysis, training, tuning, evaluation,
> saved-model inference, and the final interface are planned in the [roadmap](docs/roadmap.md).
> No trained models or performance results are available yet.

## Quick start

### 1. Set up locally

You need **Git** and **Python 3.12–3.14**. Python **3.12** is the preferred team baseline.
Start with a CPU environment; a GPU is not required for setup.

**Windows · PowerShell**

```powershell
git clone https://github.com/OMZaky/hotel-booking-intelligence.git
cd hotel-booking-intelligence

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m hotel_ml doctor
```

If you already have the repository, open its folder and skip the clone commands.
If you use Python 3.13 or 3.14, replace `py -3.12` with the corresponding version.

<details>
<summary><strong>Linux / macOS setup</strong></summary>

```bash
git clone https://github.com/OMZaky/hotel-booking-intelligence.git
cd hotel-booking-intelligence

python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m hotel_ml doctor
```

Use your installed supported Python version if it differs from 3.12.
For later commands, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

</details>

The explicit interpreter paths run commands in the project environment without requiring
shell activation. Installation includes the package, ML libraries, Ruff, and notebook support.

### 2. Open in VS Code

1. Open the repository folder in VS Code.
2. Install the extensions suggested by the workspace.
3. Run **Python: Select Interpreter** and select the interpreter inside `.venv`.

The workspace includes debugging entries for environment inspection and run initialization,
plus a **Hotel ML: tests** task. See [the checks below](#development-checks) before contributing.

### 3. Add the dataset

Place your supplied file at **`data/raw/H2.csv`**. Setup does not download it.

| Expected source | Details |
| --- | --- |
| Dataset | Hotel Booking Demand — **City Hotel (H2) only** |
| Original size | 79,330 bookings and 31 source variables |
| Arrival period | July 2015–August 2017 |
| Variable definitions | [Original dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) |

The file must be inspected and validated before training. Do not combine H1 and H2.
If your copy uses renamed columns, document a one-to-one mapping to the original source names.
Raw data is ignored by Git; keep the original file unchanged.
See [dataset instructions](data/README.md) for details.

The `doctor` command reports the expected data path, file availability, Python version,
and installed ML dependencies. A missing dataset is expected until you supply H2.csv.

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OMZaky/hotel-booking-intelligence/blob/main/notebooks/colab_bootstrap.ipynb)

Open [`notebooks/colab_bootstrap.ipynb`](notebooks/colab_bootstrap.ipynb), then:

1. Choose a **CPU** runtime initially.
2. Set `REVISION` to the commit SHA you want to run. The `main` default is useful for setup.
3. Run the clone/install cells; they install the same shared package used locally.
4. For Drive-backed data and outputs, place H2.csv at
   `MyDrive/hotel-booking-intelligence/data/H2.csv` and run the optional Drive cell.
5. Run the experiment-initialization cell.

The Drive cell copies input data into the runtime and writes run records to persistent Drive
storage. Files left only inside a Colab VM can disappear when the runtime ends.

**The launcher currently prepares the environment and records experiment metadata; it does
not train models.** A live end-to-end Colab run remains to be verified.
See [the Colab guide](docs/colab.md) for persistence and runtime details.

## Experiments

An **experiment recipe** describes what to run. A **run** records one execution of that recipe.
The same recipe can produce multiple independent runs.

| Starter recipe | Config |
| --- | --- |
| Regression · Ridge | [ridge-baseline.json](configs/regression/v1/ridge-baseline.json) |
| Classification · Logistic regression | [logistic-regression-baseline.json](configs/classification/v1/logistic-regression-baseline.json) |
| Clustering · K-Means | [kmeans-baseline.json](configs/clustering/v1/kmeans-baseline.json) |

These recipes are setup examples. Their feature policies and data splits still need review
before they can become training protocols.

**Validate a recipe and initialize a run:**

```powershell
.\.venv\Scripts\python.exe -m hotel_ml validate-config --config configs/regression/v1/ridge-baseline.json
.\.venv\Scripts\python.exe -m hotel_ml init-run --config configs/regression/v1/ridge-baseline.json
```

Once H2.csv is available, add `--data data/raw/H2.csv` to `init-run` to record its SHA-256 hash.
Use `--help` with the CLI or any subcommand to inspect available options.

Each run creates:

```text
artifacts/<task>/<version>/<unique-run-id>/
├── config.json       # Recipe used for this execution
└── manifest.json     # Code commit, dirty state, environment, and optional data hash
```

Initialization records status `initialized`; it does not fit a model.
Use `--output-root` or `HOTEL_ML_ARTIFACT_ROOT` to choose persistent output storage.
Generated artifacts are ignored by Git.

Keep recipe versions under `configs/<task>/<version>/` and share implementation in
`src/hotel_ml/`. Introduce a new recipe version when the scientific protocol changes;
create a new run ID for each execution. See [experiment conventions](configs/README.md).

### Evaluation commitments

- Define the prediction point and task-specific feature availability **before training**.
  The published data contains many snapshots relative to arrival; booking-creation availability
  cannot be assumed. See the [preliminary leakage audit](docs/leakage-audit.md).
- Fit imputation, encoding, scaling, feature selection, PCA, and LDA inside training folds.
  Keep the final supervised test set out of model selection.
- Report baseline and tuned comparisons for every selected algorithm, with search spaces,
  selection criteria, validation results, and training time.
- Include MAE, MSE and/or RMSE, and R-squared for regression; accuracy, precision, recall, F1,
  confusion matrices, and ROC-AUC and/or PR-AUC for classification; and Silhouette and
  Davies-Bouldin scores with interpretable profiles for clustering.
- Include an ensemble/boosting model among the supervised models and a kernel-based
  classification experiment.
- Compare at least two supervised models with and without PCA, and original/PCA/LDA
  representations for at least one classifier.
- Record code, data, configuration, seeds, and environment for reproducibility.

Dependencies currently use compatibility bounds rather than a resolved lock.
A tested dependency lock is still required before reported experiments.
The complete assessment requirements are in the [project brief](docs/CSE382_Project_Description_Milestones.pdf).

## Project structure

The repository keeps only folders used by the current setup:

```text
hotel-booking-intelligence/
├── configs/              # Versioned experiment recipes for the three tasks
├── data/raw/             # Your H2.csv goes here; data is ignored by Git
├── src/hotel_ml/
│   ├── experiments/      # Recipe validation and run records
│   └── cli.py            # Shared command-line entry point
├── notebooks/            # Colab launcher
├── artifacts/            # Ignored generated outputs
├── tests/                # Automated checks
├── scripts/              # Repository validation utility
└── docs/                 # Brief, setup guides, scientific decisions, and roadmap
```

Shared VS Code settings live in `.vscode/`; automated GitHub checks live in `.github/`.
Add new folders only when real code or outputs need them. The
[future-structure note](docs/future-structure.md) records the deferred layout.

## Development checks

Run these from the repository root before committing:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/validate_repository.py
```

To apply formatting, run `python -m ruff format .` with the project environment's interpreter.
Import-order fixes are separate: `python -m ruff check --select I --fix .`.
GitHub Actions runs lint, formatting, tests, recipe/notebook validation, and dependency-import
checks on **Windows and Ubuntu with Python 3.12 and 3.14**.

## Milestones

| Delivery | Date | Expected outcome |
| --- | --- | --- |
| **Milestone 1** | November 5, 2026 | Data audit, reusable preprocessing, one initial model per task, preliminary results, and progress report |
| **Milestone 2** | December 11, 2026 | Complete comparisons and tuning, PCA/LDA, saved-pipeline application, and final report |
| **Live discussion** | December 18–24, 2026 | Working demonstration and individual understanding of the complete workflow |

See the [phased roadmap](docs/roadmap.md) for working targets and the midterm buffer.

This is a **five-member team project**. Each member contributes meaningful commits from their
own GitHub account and must understand the complete workflow. Record identities and
responsibilities in [TEAM.md](TEAM.md).

For each submission, package the source, configs, report, and reviewed results; include
[this repository's URL](https://github.com/OMZaky/hotel-booking-intelligence) and the submitted
commit SHA. Keep fitted pipelines out of the ZIP, but retain them locally for the live demonstration.

**Submitted commits:** Milestone 1 — pending · Milestone 2 — pending.

## Documentation

| Guide | What you will find |
| --- | --- |
| [Project brief](docs/CSE382_Project_Description_Milestones.pdf) | Course requirements and grading |
| [Roadmap](docs/roadmap.md) | Delivery phases and open decisions |
| [Architecture](docs/architecture.md) | Current implementation boundaries and versioning |
| [ML practices](docs/ml-practices.md) | Evaluation, preprocessing, and reproduction conventions |
| [Leakage audit](docs/leakage-audit.md) | Feature timing risks and review checklist |
| [Colab guide](docs/colab.md) | Remote execution and output persistence |
| [Future structure](docs/future-structure.md) | Folders to introduce when needed |
| [References](docs/references.md) | Primary technical and dataset sources |

Dataset reference: António, Almeida, and Nunes, *Hotel booking demand datasets*,
Data in Brief, Volume 22 (2019), pp. 41–49.
