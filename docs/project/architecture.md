# Repository design

Status: bootstrap defaults selected by the assistant from the project brief; scientific choices below remain proposals for discussion.

## Purpose and constraints

Hotel Booking Intelligence is a Python application for City Hotel (H2) ADR regression, cancellation classification, and unsupervised booking segmentation. The source of course requirements is `docs/reference/CSE382_Project_Description_Milestones.pdf`. Use H2 only; the expected original dataset has 79,330 rows and 31 variables. The user will supply the dataset. No dataset is downloaded by setup.

The final solution must run outside notebooks, compare at least three models per task before and after tuning, include PCA/LDA comparisons, and load saved pipelines for inference. Milestone 1 is 2026-11-05; Milestone 2 is 2026-12-11; discussion is 2026-12-18 through 2026-12-24.

## Approaches considered

| Approach | Benefit | Cost | Decision |
| --- | --- | --- | --- |
| Separate notebooks for every model/version | Fast first exploration | Duplicated logic, drifting splits, difficult application reuse | Exploration only |
| Shared Python package, declarative experiments, thin notebooks | Same code locally and in Colab; auditable comparisons | Small initial setup cost | Recommended bootstrap default |
| Full orchestration platform with hosted tracking and data remotes | Multi-user operations and dashboards | Accounts, credentials, storage maintenance disproportionate to this project | Revisit only if needed |

## Current boundaries

- `src/hotel_ml/experiments/`: config validation and independent run records.
- `src/hotel_ml/cli.py`: environment inspection, recipe validation and run initialization.
- `notebooks/`: the Colab launcher calling the shared package.
- `configs/`: versioned task recipes; `data/raw/`: the user-supplied H2 location.
- `artifacts/`: ignored generated outputs; `tests/` and `scripts/`: development checks.

Unused package, application, derived-data and reporting folders have been removed to keep
onboarding straightforward. Their responsibilities and creation triggers are recorded in
[future structure](future-structure.md). Add them alongside real code or outputs.
Only config validation, run initialization and environment inspection are implemented now.

## Versions and run identity

An experiment is a scientific recipe under `configs/<task>/<version>/`. A run is one execution of that recipe, with a unique UTC/UUID identifier under `artifacts/<task>/<version>/<run_id>/`. Model family, feature representation, baseline/tuned stage, seed, and split identifier are separate config fields. A future v2 need not duplicate v1 source code. Retain old recipes and identify the source commit in every run.

Run initialization captures the resolved configuration, code SHA and dirty state, installed package versions, runtime/platform, and dataset SHA-256 when supplied. Its status is `initialized`, never `trained`. Successful training will add pipeline artifacts, metrics, timings, search results, and a completed status in later phases. Create `reports/` when reviewed publishable results exist; `artifacts/` contains ignored generated outputs.

## Environment

Python 3.12 is the preferred local baseline. The package permits 3.12–3.14 to accommodate managed Colab runtimes and the existing local Python 3.14 installation. CI targets 3.12 and 3.14. Core setup commands use the standard library; ML and development dependencies are optional extras. Dependencies have compatibility bounds; a tested resolved lock is still pending. Resolve and commit a tested environment lock before producing reported experiments. Always record the actual environment per run.

Use scikit-learn/pandas/NumPy first; deep learning and GPUs are not prerequisites for this tabular assignment. Start with local JSON records; MLflow is an optional later tracking interface if comparison volume warrants it. Do not add paid services or external credentials just for setup.

## Scientific choices still open

- Define a prediction point for each supervised task before approving any feature allowlist. The source paper describes snapshots relative to the day before arrival, which does not establish booking-creation availability.
- Freeze supervised holdouts before model selection. Compare a time-aware design with a stratified random benchmark only after examining date coverage, duplicates, and label availability. Do not claim an arrival-date split simulates booking-creation deployment without further justification.
- Candidate families: Ridge/linear regression, random forest, gradient boosting; logistic regression, random forest, kernel SVM; K-Means, GMM, and sampled agglomerative clustering. These are proposals, not a locked model list.
- Choose a primary regression score and classification operating criterion; record all rubric-required metrics.
- CLI first; a Streamlit presentation layer is a proposed later addition.
- The team has five students; record identities and roles in TEAM.md.

## GitHub and collaboration

Public repository target: OMZaky/hotel-booking-intelligence. Public means anyone can read;
only the owner and explicitly added collaborators can write. Five human members will each
make meaningful identifiable contributions. Keep data, credentials and fitted binaries outside Git.
No software license is assumed for the supplied course PDF.
