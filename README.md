# Hotel Booking Intelligence

A reproducible CSE382 project for **City Hotel (H2)** booking analysis:
ADR regression, cancellation classification, and unsupervised segmentation.

Repository: [OMZaky/hotel-booking-intelligence](https://github.com/OMZaky/hotel-booking-intelligence).

**Status:** repository bootstrap. Config validation, environment inspection, and run
initialization work. Data loading, training, tuning, evaluation, inference, and the final UI
are scheduled in the [roadmap](docs/roadmap.md). No dataset or model results are included.

## VS Code setup

Python **3.12** is the preferred baseline. The package supports 3.12–3.14;
the current workstation has 3.14. From the repository root, Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m hotel_ml doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If only 3.14 is installed, use `py -3.14 -m venv .venv` instead. Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m hotel_ml doctor
.venv/bin/python -m unittest discover -s tests -v
```

Install recommended VS Code extensions. Use **Python: Select Interpreter** to choose
`.venv`. Run and Debug includes doctor and run-initialization entries.
The **Hotel ML: tests** task runs the tests.

Offline bootstrap checks with an existing Python 3.12–3.14 interpreter:

```powershell
$env:PYTHONPATH = "src"
python -m hotel_ml doctor
python -m unittest discover -s tests -v
python scripts/validate_repository.py
```

Dependencies currently have compatibility ranges, **not a resolved lock**. Installation and
lock generation require network access. Before reported experiments, commit a tested resolved
lock; each run also records actual installed versions. Scientific dependencies are optional
for the bootstrap commands.

## Data and experiments

Provide `data/raw/H2.csv`; see [data instructions](data/README.md). No automatic download.

```bash
python -m hotel_ml validate-config --config configs/regression/v1/ridge-baseline.json
python -m hotel_ml init-run --config configs/regression/v1/ridge-baseline.json
# Once H2.csv is available, record its exact hash:
python -m hotel_ml init-run --config configs/regression/v1/ridge-baseline.json --data data/raw/H2.csv
```

`init-run` creates metadata only, with status `initialized`. It does not train.
Each execution gets `artifacts/<task>/<version>/<run_id>/` containing config, code SHA/dirty flag,
runtime/package versions and optionally a dataset hash. Set `HOTEL_ML_ARTIFACT_ROOT` or
`--output-root` to use persistent storage.

## Google Colab

[Open the launcher](https://colab.research.google.com/github/OMZaky/hotel-booking-intelligence/blob/main/notebooks/colab_bootstrap.ipynb)
or manually upload `notebooks/colab_bootstrap.ipynb`.
It clones a selected revision, installs this package, copies user data from Drive, and writes
run outputs to Drive. See [Colab instructions](docs/colab.md).

## Structure

Only folders used by the current setup are present:

```text
configs/<task>/<version>/   # Experiment recipes
data/raw/                   # Place your H2.csv here
src/hotel_ml/
  experiments/              # Config validation and run provenance
  cli.py                    # Shared command-line entry point
notebooks/                  # Colab launcher
artifacts/                  # Ignored generated run records
tests/                      # Automated checks
scripts/                    # Repository validation utility
docs/                       # Project brief, instructions and roadmap
.github/                    # CI and review template
.vscode/                    # Editor tasks and debugging
```

Add new folders as their implementation or outputs become necessary; the
[future structure note](docs/future-structure.md) records their intended roles.
See [recipe versioning](configs/README.md), [architecture](docs/architecture.md),
[ML practices](docs/ml-practices.md), [leakage audit](docs/leakage-audit.md)
and [sources](docs/references.md).

## Delivery and collaboration

- **November 5, 2026:** prepared data, shared preprocessing, initial models and progress report.
- **December 11, 2026:** full comparisons/tuning, PCA/LDA, application and final report.
- **December 18–24, 2026:** live demonstration and individual discussion.

Five team members will contribute meaningful commits from their own accounts.
Record identities in [TEAM.md](TEAM.md). Preserve real authorship and review focused branches.
No license is assumed for the supplied course PDF.

For each milestone release, record the submitted commit SHA here and include this GitHub URL.
A submission SHA does not exist yet. ZIP source, configs, reports and reviewed results;
exclude fitted model binaries but retain them locally for the live demonstration.
