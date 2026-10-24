# Repository design

Status: bootstrap defaults selected by the assistant from the project brief; scientific choices below remain proposals for discussion.

## Purpose and constraints

Hotel Booking Intelligence is a Python application for City Hotel (H2) ADR regression, cancellation classification, and unsupervised booking segmentation. The source of course requirements is `docs/CSE382_Project_Description_Milestones.pdf`. Use H2 only; the expected original dataset has 79,330 rows and 31 variables. The user will supply the dataset. No dataset is downloaded by setup.

The final solution must run outside notebooks, compare at least three models per task before and after tuning, include PCA/LDA comparisons, and load saved pipelines for inference. Milestone 1 is 2026-11-05; Milestone 2 is 2026-12-11; discussion is 2026-12-18 through 2026-12-24.

## Approaches considered

| Approach | Benefit | Cost | Decision |
| --- | --- | --- | --- |
| Separate notebooks for every model/version | Fast first exploration | Duplicated logic, drifting splits, difficult application reuse | Exploration only |
| Shared Python package, declarative experiments, thin notebooks | Same code locally and in Colab; auditable comparisons | Small initial setup cost | Recommended bootstrap default |
| Full orchestration platform with hosted tracking and data remotes | Multi-user operations and dashboards | Accounts, credentials, storage maintenance disproportionate to this project | Revisit only if needed |

## Boundaries

- `src/hotel_ml/data/`: loading, source-schema checks, immutable split manifests.
- `src/hotel_ml/features/`: feature availability policies and reproducible feature builders.
- `src/hotel_ml/preprocessing/`: fitted transformations composed inside model pipelines.
- `src/hotel_ml/tasks/{regression,classification,clustering}/`: task-specific model definitions and searches.
- `src/hotel_ml/evaluation/`: metrics, comparisons, and final evaluation protocol.
- `src/hotel_ml/experiments/`: config validation and independent run directories.
- `src/hotel_ml/inference/`: loading saved pipelines and validating prediction inputs.
- `apps/`: future UI; the CLI remains usable independently.
- `notebooks/`: exploration and Colab launchers calling package code.

Only config validation, run initialization, and environment inspection are implemented in the bootstrap. Empty package boundaries reserve the remaining work; they do not claim working model training.

## Versions and run identity

An experiment is a scientific recipe under `configs/<task>/<version>/`. A run is one execution of that recipe, with a unique UTC/UUID identifier under `artifacts/<task>/<version>/<run_id>/`. Model family, feature representation, baseline/tuned stage, seed, and split identifier are separate config fields. A future v2 need not duplicate v1 source code. Retain old recipes and identify the source commit in every run.

Run initialization captures the resolved configuration, code SHA and dirty state, installed package versions, runtime/platform, and dataset SHA-256 when supplied. Its status is `initialized`, never `trained`. Successful training will add pipeline artifacts, metrics, timings, search results, and a completed status in later phases. `reports/` contains curated publishable results; `artifacts/` contains ignored generated outputs.

## Environment

Python 3.12 is the preferred local baseline. The package permits 3.12–3.14 to accommodate managed Colab runtimes and the existing local Python 3.14 installation. CI targets 3.12 and 3.14. Core setup commands use the standard library; ML and development dependencies are optional extras. Dependencies initially have compatibility bounds, not a resolved lock: network restrictions prevent generating and validating a lock in this session. Resolve and commit a tested environment lock before producing reported experiments. Always record the actual environment per run.

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
