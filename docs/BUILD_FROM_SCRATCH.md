# Build Hotel Booking Intelligence from scratch

A hands-on, self-contained guide for a new repository. Prepared against the CSE382 project brief on **28 October 2026**.

You will write the project yourself. This guide does not depend on the Python code, configuration loader, or Colab notebook in the repository where you found it. You may copy this guide into your new repository's `docs/` folder. Get the project brief from your course and keep your own copy beside it.

**No dataset is downloaded or supplied by this guide.** You will provide H2.csv. Examples using synthetic data teach mechanics; their outputs are not hotel-model results.

For a deeper treatment of cleaning and feature design, see the companion
[data preparation and feature engineering handbook](DATA_PREPARATION_AND_FEATURE_ENGINEERING.md).

## How to use this guide

For each chapter:

1. Read the explanation before typing the commands.
2. Type the small examples yourself, run them, and inspect their outputs.
3. Implement the exercise in your own words.
4. Complete the checkpoint and answer the understanding questions.
5. Make a meaningful Git commit when a coherent piece works.

Keep a learning journal: what you expected, what happened, what failed, and what you changed. If you cannot explain a line, pause and investigate it. Copying a large finished script makes that harder.

Commands labeled **proposed interface** are commands you will implement later; they do not exist in a fresh repository. Code labeled **example** is a focused illustration, not a finished application. Code examples in a section can depend on variables introduced earlier in that same section.

### Reading map

- [1. What you are building](#1-what-you-are-building)
- [2. Choose your learning and implementation route](#2-choose-your-learning-and-implementation-route)
- [3. Tools, packages, and installation](#3-tools-packages-and-installation)
- [4. Create a genuinely new project](#4-create-a-genuinely-new-project)
- [5. Git and GitHub from the first day](#5-git-and-github-from-the-first-day)
- [6. Make an importable Python package](#6-make-an-importable-python-package)
- [7. Understand Python and ML objects](#7-understand-python-and-ml-objects)
- [8. Load and validate H2](#8-load-and-validate-h2)
- [9. Define the prediction question and audit leakage](#9-define-the-prediction-question-and-audit-leakage)
- [10. Freeze splits before modeling](#10-freeze-splits-before-modeling)
- [11. Explore training data](#11-explore-training-data)
- [12. Build preprocessing and features](#12-build-preprocessing-and-features)
- [13. Your first complete learning example](#13-your-first-complete-learning-example)
- [14. Regression experiments](#14-regression-experiments)
- [15. Classification experiments](#15-classification-experiments)
- [16. Cross-validation and hyperparameter tuning](#16-cross-validation-and-hyperparameter-tuning)
- [17. PCA and LDA](#17-pca-and-lda)
- [18. Clustering](#18-clustering)
- [19. Configs, versions, and run records](#19-configs-versions-and-run-records)
- [20. Colab and your compute budget](#20-colab-and-your-compute-budget)
- [21. Save pipelines and implement inference](#21-save-pipelines-and-implement-inference)
- [22. Build the application](#22-build-the-application)
- [23. Test and automate checks](#23-test-and-automate-checks)
- [24. Compare, select, and evaluate once](#24-compare-select-and-evaluate-once)
- [25. Report, package, and demonstrate](#25-report-package-and-demonstrate)
- [26. Debugging reference](#26-debugging-reference)
- [27. Completion and understanding checklists](#27-completion-and-understanding-checklists)
- [28. Primary-source reading list](#28-primary-source-reading-list)

## 1. What you are building

The course project is a complete ML application, not just a collection of trained estimators.

~~~text
user-provided H2.csv
    -> schema and quality checks
    -> declared task, prediction point, feature policy, and split protocol
    -> training-only fitted preprocessing
    -> baseline models
    -> training/CV-based tuning and representation experiments
    -> comparisons and frozen model selection
    -> final supervised test evaluation
    -> saved complete inference pipelines
    -> CLI or graphical application
    -> report, source ZIP, and live demonstration
~~~

Clustering is a related, genuinely unsupervised analysis. It has its own feature policy, fitting cohort, optimization criteria, and interpretation. Do not give it cancellation labels and then describe the resulting groups as independently discovered.

### 1.1 Requirement map

| Requirement | What you must produce |
| --- | --- |
| Data | H2 only; original expected size 79,330 rows, 31 source variables; verify the actual file |
| Regression | Predict ADR; at least three model families, each baseline and tuned |
| Classification | Predict IsCanceled; at least three models, each baseline and tuned; include a kernel experiment |
| Ensemble | At least one ensemble/boosting model among supervised experiments |
| Clustering | At least three algorithms, each baseline and systematically optimized |
| PCA | Explain retained variance; compare at least two supervised models with and without PCA |
| LDA | At least one classifier compared on original, PCA, and LDA representations |
| Evaluation | Required metrics, validation/CV evidence, fair comparisons, and final supervised test results |
| Leakage | Dedicated audit covering input timing, transformations, splits, tuning, and outcomes |
| Inference | Load saved preprocessing/model pipelines; do not retrain on application startup |
| Application | Analysis, training or artifact loading, comparisons, tuning access, ADR/cancellation predictions, clustering views, and selected-model information |
| Team | Four to five students under the brief; your intended team has five; identifiable contributions |
| Reproduction | Dependencies, exact commands, seeds, data/split identity, source version, and available local artifacts |
| Submission | Source ZIP, reports, figures/tables, GitHub URL and submitted SHA; exclude fitted model binaries from the ZIP |

A dummy predictor is an additional sanity check, not a substitute for the three substantive models. If you select an extra fourth model, include its baseline and tuned/optimized results too. Avoid adding more algorithms than you can evaluate and explain properly.

### 1.2 Milestones and learning priorities

- **5 November 2026:** prepared data, EDA, reusable preprocessing, one initial model per task, validation results, preliminary leakage audit, contribution record, and progress report. Keep supervised final tests sealed.
- **6-12 November:** midterms; the brief schedules no milestone or mandatory project discussion here.
- **11 December:** complete final project submission.
- **18-24 December:** final demonstration and individual discussion.

A practical plan from 28 October: tools/package/Git and data inspection first; splits and pipelines next; one Ridge, logistic-regression, and K-Means baseline next; report and reproduction by 4 November. Then expand families, tuning, representations, and application after the initial milestone. Change the plan if an audit exposes invalid assumptions; do not rush invalid results into a report.

The grading is discussion 40%, report 20%, technical work 40%. You need to understand all parts, even when teammates divide implementation work.

## 2. Choose your learning and implementation route

### 2.1 What "from scratch" means here

You can personally author the data loading, experiments, evaluation, application, tests, and documentation while using scikit-learn's algorithm implementations. The brief explicitly permits and expects library implementations.

Writing gradient descent, tree splitting, or SVM optimization yourself is an optional learning exercise. It is a separate undertaking from building the complete system. A small manual linear-regression exercise is valuable; nine hand-written algorithms are not required for this submission.

### 2.2 Development routes

| Route | How it works | Benefit | Tradeoff |
| --- | --- | --- | --- |
| Package and CLI first | Write reusable functions in a Python package and call them from scripts/notebooks | Clear ownership, reproducible commands, direct path to the application | Learn imports and packaging early |
| Notebook discovery, then extract | Explore briefly in a notebook; move successful logic into modules after each small experiment | Convenient inspection and visualization | Extraction is mandatory; cell state can hide dependencies |
| UI first | Build a Streamlit screen, then connect a package | Early visual feedback | Easy to mix UI, model fitting, and mutable state |

**Recommended learning route:** package and CLI first, with small notebooks for plots and Colab execution. A complete CLI already satisfies the interface option in the brief. Add a graphical layer after the reusable functions work.

### 2.3 Keep choices explicit

Keep `docs/decisions.md` with one small entry per real decision:

- question and assumed prediction point;
- alternatives;
- choice and rationale;
- evidence or limitation;
- date and source commit.

Do not build every alternative in this guide. Pick one route at each fork, understand it, and keep the others as informed options.

## 3. Tools, packages, and installation

### 3.1 Programs and accounts

| Item | Role | Needed now? |
| --- | --- | --- |
| Python 3.12 | Runs Python code; includes the standard library and normally pip | Yes; preferred baseline |
| Python 3.13/3.14 | Alternative supported runtime for the examples' general approach | Possible; check dependency compatibility and record the actual version |
| Git | Local snapshots, history, branches, merges | Yes |
| GitHub account | Hosts the shared repository and reviews | Yes |
| VS Code | Editor, debugger, terminal, and notebook UI | Recommended |
| VS Code Python extension | Interpreter selection, Python run/debug support | Recommended |
| Pylance | Type information and editor diagnostics | Recommended |
| Jupyter extension | Notebook UI inside VS Code | If using notebooks locally |
| Ruff extension | Runs the formatter/linter from the project environment | Recommended |
| GitHub CLI, `gh` | Convenient login, repository creation, and PR commands | Optional; browser GitHub also works |
| Google account and Colab | Remote notebooks and compute | Yes for your intended workflow |

Use official installers: [Python](https://www.python.org/downloads/), [Git](https://git-scm.com/downloads), [VS Code](https://code.visualstudio.com/), and optionally [GitHub CLI](https://cli.github.com/). You do not need Docker, Node.js, a database, CUDA drivers, or an MLOps server for the first working model.

Verify installations in a new PowerShell terminal:

~~~powershell
py --version
py -0p
git --version
~~~

`py` is the Windows launcher. `py -0p` lists detected Python installations and paths. If your installation uses the newer Python manager, consult `py --help` for its listing command. On Linux/macOS, use `python3 --version` and the relevant package/installer instructions.

### 3.2 Python packages and why they exist

| Package | Import | What it contributes |
| --- | --- | --- |
| NumPy | `numpy` | Arrays, vectorized arithmetic, random generators, and numerical data representation |
| pandas | `pandas` | DataFrames, CSV loading, column handling, grouping, dates, inspection |
| SciPy | `scipy` | Scientific routines, sparse matrices, distributions, hierarchical linkage/dendrogram support |
| scikit-learn | `sklearn` | Estimators, preprocessing, pipelines, CV, tuning, PCA/LDA, clustering, metrics |
| Matplotlib | `matplotlib.pyplot` | Base plotting tools and explicit figure control |
| seaborn | `seaborn` | Statistical plots built on Matplotlib; convenient rather than mandatory |
| joblib | `joblib` | Persisting trusted fitted sklearn pipelines and arrays |
| Ruff | command or `python -m ruff` | Formatting and static checks; does not prove scientific correctness |
| ipykernel | `ipykernel` | Makes your Python environment available as a notebook execution kernel |
| setuptools | build backend | Builds/installs your own Python package |
| pip | `python -m pip` | Installs packages into the Python interpreter you selected |

These standard-library modules need **no pip installation**: `pathlib`, `argparse`, `json`, `hashlib`, `datetime`, `logging`, `time`, `subprocess`, `tempfile`, `unittest`, `dataclasses`, and `importlib.metadata`.

Optional packages should be installed only when you choose their route:

| Optional package | Why choose it? | Cost or caution |
| --- | --- | --- |
| XGBoost | Additional boosted-tree family; CPU or explicitly configured GPU execution | New dependency and hardware/version checks |
| Optuna | Adaptive hyperparameter searches | More moving parts; objective must still respect CV and holdouts |
| Streamlit | Fast Python graphical application | Learn reruns/session state; keep fitting outside ordinary page rerenders |
| Gradio | Alternative prediction/analysis interface | Still needs the full rubric's analysis/tuning/comparison features |
| FastAPI + Uvicorn | API service with explicit input schemas | HTTP, schemas, deployment and possibly frontend work |
| pytest | Alternative test runner and fixtures | `unittest` is sufficient; choose a testing style deliberately |
| JupyterLab | Standalone local notebook application | Unnecessary if VS Code notebooks are sufficient |
| MLflow | Searchable experiment tracking and artifact metadata | Optional after understanding plain run records |
| PyYAML | YAML configuration files | JSON needs no extra dependency; use safe loading for YAML |
| imbalanced-learn | Resampling inside compatible pipelines | Misplaced resampling leaks; class weights are often a simpler first route |

Do not install TensorFlow/PyTorch solely because the assignment says "machine learning." Add them if you deliberately choose a neural-network experiment and can support its tuning, persistence, inference, and explanation.

### 3.3 Installation concepts

- An **interpreter** is the Python executable running a command.
- An **environment** is the interpreter plus its installed packages.
- A **dependency specification** states what versions may be installed.
- A **resolved environment record/lock** states the versions actually selected.
- A **kernel** is the process executing notebook code. It can differ from the terminal's Python.

Using `python -m pip` connects pip to that specific Python. Calling a bare `pip` can accidentally target a different installation. Installation success does not prove imports, hardware support, or reproducibility across operating systems.

Read [Python venv](https://docs.python.org/3.12/library/venv.html) before continuing.

## 4. Create a genuinely new project

Use a new folder outside the old repository. Do not delete/reset the old repository to make an empty one. The following example name is `hotel-booking-lab`; substitute your chosen name consistently.

**Windows PowerShell:**

~~~powershell
New-Item -ItemType Directory -Force -Path (Join-Path $HOME "Projects")
Set-Location -LiteralPath (Join-Path $HOME "Projects")
New-Item -ItemType Directory -Path hotel-booking-lab
Set-Location -LiteralPath hotel-booking-lab
py -3.12 -m venv .venv
~~~

What each action does:

1. Ensures a parent directory exists; `$HOME` is read, not changed.
2. Moves the terminal into that parent.
3. Creates the new project directory. If it already exists, inspect it instead of assuming it is empty.
4. Changes the terminal's current directory.
5. Creates an isolated environment at `.venv` using Python 3.12.

Activation is optional:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Activation changes the terminal's command search path so `python` points to your environment. It does not install packages. If PowerShell refuses activation, use the explicit interpreter path below; you do not need to disable your machine's execution policy.

~~~powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
~~~

The path should end in this new project's `.venv\Scripts\python.exe`.

**Linux/macOS equivalent:**

~~~bash
mkdir -p ~/Projects
cd ~/Projects
mkdir hotel-booking-lab
cd hotel-booking-lab
python3.12 -m venv .venv
.venv/bin/python -c "import sys; print(sys.executable)"
~~~

For all later commands, **`python` means this project's interpreter**. Either activate the environment or replace it with the explicit platform-specific path. All relative paths assume the terminal is at your new repository root.

**Checkpoint:** Explain which Python runs your terminal and which runs a notebook. Nothing has trained yet, and you have not copied the old project's implementation.


## 5. Git and GitHub from the first day

Git is local history. GitHub hosts a copy and collaboration tools. They are separate: a commit can exist locally before you push it.

### 5.1 The four places to understand

| Place | Meaning | Useful command |
| --- | --- | --- |
| Working tree | Files you currently edit | `git diff` |
| Staging area/index | The exact changes selected for the next commit | `git diff --cached` |
| Local repository | Commits stored under `.git` | `git log --oneline` |
| Remote repository | Hosted repository such as `origin` on GitHub | `git remote -v` |

A branch is a movable name pointing to a commit. A commit records a snapshot plus parent history and author information. A pull request is a proposal to merge changes, not the same thing as a Git push.

### 5.2 Start with a README and ignore policy

Create `README.md` in the editor. Write the project's purpose, current status, and your first setup command. Improve it as the program grows.

Create `.gitignore` before adding data or environments:

~~~gitignore
.venv/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.ruff_cache/
.pytest_cache/
.ipynb_checkpoints/
.env
.env.*
!.env.example
*.pem
*.key
data/raw/*
!data/raw/.gitkeep
data/interim/
data/processed/
artifacts/
mlruns/
mlartifacts/
*.joblib
*.pkl
*.pickle
*.onnx
*.pt
*.h5
~~~

An ignore rule stops matching **untracked** files being added normally. It does not remove files already committed. Never use `git add -f` to bypass the policy casually. If a credential was committed, deleting it in a later commit does not revoke it; revoke/rotate the credential and investigate history removal before publishing further.

Create the repository and set your identity **for this repository**:

~~~powershell
git init -b main
git config user.name "YOUR ACTUAL NAME"
git config user.email "YOUR GITHUB-ASSOCIATED EMAIL"
git status
git add README.md .gitignore
git diff --cached
git commit -m "docs: start hotel booking project"
~~~

Replace the two identity placeholders. Using a verified GitHub email or your GitHub-provided no-reply email makes attribution work as intended. Repository-local identity avoids changing every other project on your computer.

- `init -b main` creates Git metadata and selects the initial branch name.
- `config` records your author identity; it is not an authentication login.
- `status` reports untracked/staged/modified files.
- `add` chooses the next snapshot.
- `diff --cached` lets you check exactly what you are committing.
- `commit -m` creates local history with a concise message explaining the change.

### 5.3 Publish the new repository

**Route A: browser.** Create a new empty GitHub repository under your account. When connecting existing local history, do not initialize a second README, license, or ignore file remotely. Then use the URL GitHub gives you:

~~~powershell
git remote add origin https://github.com/YOUR_USERNAME/hotel-booking-lab.git
git push -u origin main
~~~

Replace `YOUR_USERNAME` and the repository name. `origin` is a conventional nickname for the URL. `-u` establishes tracking so later `git push` and `git pull` know the default branch relationship.

**Route B: GitHub CLI.** Install `gh`, authenticate using the interactive browser/device flow, and create the repository:

~~~powershell
gh auth login
gh auth status
gh repo create hotel-booking-lab --public --source . --remote origin --push
~~~

Use `--private` instead if that is your visibility decision and the course permits it. Do not execute both creation routes for the same repository. Do not put access tokens into code, notebook cells, repository URLs, or commit messages.

**Route C: remote first.** Create the repository in the browser, clone it, and build inside that clone from the start. In this route cloning creates `.git` and `origin`; do not repeat `git init` or `remote add`.

### 5.4 Five-person collaboration

The owner opens GitHub repository **Settings -> Collaborators -> Add people**. Each invited person accepts using their own account. Each member then clones the shared repository and creates their own environment; environments are not shared through Git.

A normal work cycle:

~~~powershell
git switch main
git pull --ff-only
git switch -c feat/data-validation
~~~

Write one coherent change. Run checks. Inspect before committing:

~~~powershell
git status
git diff
git add .
git diff --cached
git commit -m "feat: validate H2 source columns"
git push -u origin feat/data-validation
~~~

Open a pull request against `main`. Describe what changed, why, how it was checked, and any scientific assumptions. Another member reviews it. Merge after review and successful checks.

Before the next task, return to `main` and pull. `--ff-only` refuses an unexpected merge rather than silently creating one. If it refuses, inspect `git status`, `git log --oneline --graph --all`, and the branch relationship; do not force-push to make the warning disappear.

If two people edit the same lines:

1. Save/commit your local work.
2. Update `main`.
3. Merge updated `main` into your feature branch.
4. Open each conflict, understand both changes, and write the intended combined result.
5. Remove conflict markers, rerun checks, then commit the resolution.

Rebasing is an alternative for private branches, but it rewrites commits. Learn ordinary branching/merging first. Do not rewrite shared `main` history.

Assign tasks with clear interfaces: one member's loader returns a DataFrame; another's pipeline consumes declared columns; another's evaluator consumes predictions. Rotate reviews and explanations. Do not make one member upload everyone's work under one account.

### 5.5 Optional main-branch protection

After your first CI workflow has completed, consider a branch rule/ruleset for main that requires a pull request, one teammate review, and the named CI check before merging. Keep force-push/deletion disabled for the shared branch. Choose the actual check name from a completed run so the rule does not wait for a nonexistent status. Add protection deliberately and verify that at least the intended owner/maintainer can resolve exceptional situations.

GitHub supports protected branches for public repositories on Free plans; private-repository availability depends on the plan. The settings UI may expose branch rules or rulesets. Consult [protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches). An agreed review process remains useful even if you do not configure enforcement yet.

**Checkpoint:** A teammate can clone your repository, read its purpose, and identify your real first commit. No dataset, environment, secret, or model binary is tracked.

## 6. Make an importable Python package

Create only these files now; add later modules when you implement their behavior:

~~~text
hotel-booking-lab/
  README.md
  .gitignore
  pyproject.toml
  requirements.txt
  src/
    booking_ml/
      __init__.py
      __main__.py
      cli.py
~~~

The distribution name can contain hyphens; the import package `booking_ml` uses an underscore because Python identifiers cannot contain hyphens.

### 6.1 Packaging and dependencies

Create `pyproject.toml`:

~~~toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "hotel-booking-lab"
version = "0.1.0"
description = "A hands-on H2 machine learning project"
requires-python = ">=3.12,<3.15"
dependencies = [
    "numpy>=2.2,<3",
    "pandas>=2.2,<3",
    "scipy>=1.15,<2",
    "scikit-learn>=1.6,<2",
    "matplotlib>=3.10,<4",
    "seaborn>=0.13,<1",
    "joblib>=1.4,<2",
]

[project.optional-dependencies]
dev = ["ruff>=0.9,<1"]
notebooks = ["ipykernel>=6.29,<8"]

[project.scripts]
booking-ml = "booking_ml.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
~~~

What the sections do:

- `build-system` tells pip how your own project is built. This is packaging, not model training.
- `project` declares the installable distribution and supported Python range.
- `dependencies` contains compatible ranges, not a resolved lock.
- `optional-dependencies` groups extras. `dev` installs Ruff; `notebooks` installs a kernel.
- `project.scripts` defines a terminal entry point that calls a Python function.
- `packages.find` tells setuptools where import packages live.
- Ruff's configuration declares a formatting width, Python syntax target, and lint rules.

Create `requirements.txt`:

~~~text
-e .[dev,notebooks]
~~~

This installs the local distribution in **editable** mode with both extras. Your project dependencies have one authoritative home in `pyproject.toml`; do not maintain a conflicting second list manually.

### 6.2 First working entry point

Create `src/booking_ml/__init__.py`:

~~~python
"""Reusable code for the hotel booking project."""
~~~
Create `src/booking_ml/cli.py`:

~~~python
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hotel booking learning project")
    parser.add_argument("--version", action="version", version="0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show the interpreter used by this process")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print("Python executable:", sys.executable)
        print("Python version:", sys.version.split()[0])
    return 0
~~~
Create `src/booking_ml/__main__.py`:

~~~python
from booking_ml.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
~~~
Install and verify:

~~~powershell
python -m pip install -r requirements.txt
python -m pip check
python -m booking_ml doctor
python -m booking_ml --help
~~~

- `-r` tells pip to read a requirements file.
- `-e .` installs the current project while keeping its source editable.
- Extras in brackets add optional dependency groups.
- `pip check` checks installed dependency compatibility; it does not test your model.
- `python -m booking_ml` executes the package's `__main__.py`.
- `main()` returns an exit code; zero means success. A failing CLI should return a nonzero code.

The console script `booking-ml doctor` is equivalent when the environment's scripts directory is on PATH. The `python -m` form makes interpreter choice clearer.

### 6.3 Verify imports and notebook selection

~~~powershell
python -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn, joblib; print('Imports OK')"
python -m ipykernel install --user --name hotel-booking-lab --display-name "Hotel booking lab"
~~~

The second command registers a kernel specification pointing to this environment. In VS Code, choose the environment in **Python: Select Interpreter** and select the matching notebook kernel. If you later delete/recreate the environment, verify the kernel still points to the correct interpreter.

### 6.4 Freeze what actually worked

After checking a clean environment:

~~~powershell
python -m pip freeze --exclude-editable > requirements-lock-windows-py312.txt
python -m pip check
~~~

This is an exact **platform/runtime-specific package snapshot**, not a universal cross-platform lock. It excludes the editable project's machine-specific path. Record the OS and Python version. Verify the text file is saved as UTF-8/ASCII; Windows PowerShell redirection can choose UTF-16, so resave with UTF-8 if necessary.

Reproduction on a matching platform:

~~~powershell
python -m pip install -r requirements-lock-windows-py312.txt
python -m pip install --no-deps -e .
python -m pip check
~~~

`--no-deps` here means "use the dependencies already installed from the tested record." It is not a way to ignore missing dependencies.

Alternatives:

- **uv:** resolves a `uv.lock`, offers fast environment management and cross-platform resolution; learn its official project workflow before adopting it.
- **pip-tools:** compiles declared inputs into pinned requirements, with environment/platform considerations.
- **Conda:** useful when managing non-Python scientific libraries; environments and solver behavior differ from pip.
- **Container image:** later deployment option; does not automatically prove data/code reproducibility and is unnecessary for your first model.

Do not combine all four. Whichever route you choose, record actual installed versions inside each experimental run. Colab may need a separate tested environment record because its Python/platform differs.

**Checkpoint:** Your own new package's doctor and help commands work from a fresh terminal. Explain how installation makes `import booking_ml` possible despite the source living under `src/`.

## 7. Understand Python and ML objects

Before loading H2, be able to explain:

- A **module** is a Python file; a **package** groups importable modules.
- A **function** takes inputs, computes behavior, and returns outputs.
- A **class** defines objects with state and methods.
- A **dictionary** maps names to values; JSON configs become dictionaries.
- A **DataFrame** is a table with column names and row indices.
- `X` is the predictor table; `y` is the target vector.
- A **parameter learned by fitting** includes a coefficient, imputation median, or tree split.
- A **hyperparameter** is supplied before fitting, such as Ridge's alpha or forest depth.
- `fit(X, y)` learns state; `transform(X)` applies a fitted transformation; `predict(X)` produces outputs.
- `fit_transform` both learns and applies. Never use it to refit preprocessing on a supervised holdout.
- A **Pipeline** chains steps in sequence. A **ColumnTransformer** applies different steps to different column groups.
- `random_state` controls specific stochastic operations; it is not a universal reproducibility switch.
- A metric measures an aspect of predictions, not whether the whole experiment is scientifically valid.

An unfitted estimator is a recipe object. After `fit`, learned sklearn attributes commonly end with an underscore, such as `classes_` or `coef_`. CV/search tools clone estimators so each fold starts unfitted. Reusing a fitted object across independent experiments without understanding its state is a common source of errors.

For DataFrames, `.iloc` selects by integer row position, while `.loc` selects index labels. Persisted split positions should be applied with `.iloc` to the exact validated source ordering.

**Practice:** Create a ten-row DataFrame, select two columns as X, select one as y, print their shapes, and deliberately trigger a missing-column error. Explain why a target included in X makes a prediction experiment invalid.


## 8. Load and validate H2

Create `data/raw/` when you receive the data. Put your original file at `data/raw/H2.csv`; create an empty `.gitkeep` if you want the otherwise empty directory tracked. Never edit the original CSV to "fix" it in place.

Add `src/booking_ml/data.py` when you implement loading. A module is enough initially; you do not need a whole directory for one function.

### 8.1 Read the file before trusting it

~~~python
from pathlib import Path

import pandas as pd

source = Path("data/raw/H2.csv")
if not source.is_file():
    raise FileNotFoundError(f"Supply H2.csv at {source.resolve()}")

df = pd.read_csv(
    source,
    na_values=["NULL"],
    skipinitialspace=True,
    dtype={"Agent": "string", "Company": "string"},
)
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
~~~
`Path` handles paths. `read_csv` parses records into a DataFrame. `na_values` adds the literal source NULL marker to pandas' missing-value handling. `skipinitialspace` removes spaces immediately after delimiters; it is not comprehensive string normalization. The dtype dictionary prevents identifiers becoming quantities you later average.

Do not print every data row into public logs. Inspect small examples locally.

### 8.2 The 31-variable checklist

Use this table to check your actual source. Roles are provisional until the timing audit; a plausible booking input is not automatically valid at booking creation.

| Source name | Meaning / semantic type | Modeling consideration |
| --- | --- | --- |
| IsCanceled | Binary final cancellation outcome | Classification target; exclude from predictor/segmentation inputs |
| LeadTime | Days from booking entry to arrival | Numeric; skew and snapshot amendments need review |
| ArrivalDateYear | Arrival year | Temporal input or split/audit field, according to the question |
| ArrivalDateMonth | Arrival month | Ordered/cyclic or categorical representation |
| ArrivalDateWeekNumber | Arrival week number | Seasonal information; redundant with other dates |
| ArrivalDateDayOfMonth | Arrival day | Calendar value, not a complete date alone |
| StaysInWeekendNights | Weekend nights | Numeric; validate nonnegative values |
| StaysInWeekNights | Weekday nights | Numeric; can combine into TotalNights |
| Adults | Number of adults | Count; zero can require investigation rather than automatic deletion |
| Children | Number of children | Count; missing is not automatically zero |
| Babies | Number of babies | Count |
| Meal | Meal arrangement | Category; "Undefined" is not automatically a missing value |
| Country | Country code | Category; investigate missingness and cardinality |
| MarketSegment | Market segment | Category; review collection timing |
| DistributionChannel | Distribution channel | Category |
| IsRepeatedGuest | Repeated-guest indicator | Binary; review historical construction |
| PreviousCancellations | Previous cancellations | Historical count; verify as-of interpretation |
| PreviousBookingsNotCanceled | Previous non-canceled bookings | Historical count; verify as-of interpretation |
| ReservedRoomType | Reserved room code | Category; reserved does not mean observed unchanged at creation |
| AssignedRoomType | Assigned room code | Operational field; quarantine pending timing justification |
| BookingChanges | Accumulated amendments | Potential future information at early prediction points |
| DepositType | Deposit classification | Source-derived from payments; not automatically an initial contract input |
| Agent | Agency identifier | Categorical ID, including missing/not-applicable handling |
| Company | Company identifier | Categorical ID, often high missingness/cardinality |
| DaysInWaitingList | Time waiting for confirmation | Availability depends on operational prediction point |
| CustomerType | Booking/customer category | Category |
| ADR | Average Daily Rate | Regression target; transaction-derived timing matters |
| RequiredCarParkingSpaces | Requested parking spaces | Count whose snapshot timing needs review |
| TotalOfSpecialRequests | Number of special requests | Count whose snapshot timing needs review |
| ReservationStatus | Final reservation status | Exclude from model/segmentation predictors |
| ReservationStatusDate | Final status date | Exclude from predictors; potentially useful separately for audit/label availability |

Write a schema check that reports missing and unexpected names separately. Do not merely check that there are 31 columns: the wrong 31 columns still fail the schema.

If your file is a combined resort/city mirror, isolate City Hotel explicitly and document the source-label mapping. Check that the resulting records match H2. Never casually combine H1 and H2 to improve sample size.

### 8.3 Structural and quality checks

Implement a `validate_source(frame)` function and an audit command that reports:

- rows, columns, exact names, and data types;
- duplicate column names and duplicate records;
- missing counts and proportions;
- target validity: IsCanceled contains only 0/1 and no missing target values; ADR is numeric and finite;
- nonnegative counts/LeadTime, plausible dates, and unexpected fractional count values;
- category cardinalities and unusual category spellings;
- zero-length stays, implausible guest combinations, negative/extreme ADR;
- whether an exported index column was accidentally included.

Some checks should stop execution, such as absent targets or invalid schema. Others should produce findings requiring a documented decision, such as an extreme but potentially real ADR. Row-count differences must be investigated and explained, not hidden by sampling.

**Missing targets:** do not impute a target as if it were an input. Investigate its provenance; if records must be excluded, log the rule and resulting population.

**Duplicates:** two identical rows may represent different real bookings because identifiers were removed. Do not automatically deduplicate the entire dataset. Explain whether your goal treats them as separate bookings and how duplicate feature patterns affect split independence.

### 8.4 Normalize deliberately

Record mappings when changing source names. If trimming spaces makes two column names collide, raise an error. Preserve the mapping in metadata and the report.

Normalize identifier missing values consistently. For selected categorical columns, a useful sklearn boundary conversion is:

~~~python
import numpy as np

# Example for a previously selected list of categorical column names:
for column in categorical_columns:
    df[column] = df[column].astype(object).where(df[column].notna(), np.nan)
~~~
This is a representation conversion, not fitted preprocessing. Do not use `astype(str)` blindly: it can turn missing values into the literal category "nan".

### 8.5 Identify the exact input

~~~python
import hashlib

with source.open("rb") as stream:
    data_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
print(data_sha256)
~~~
A SHA-256 digest identifies the exact bytes. Editing whitespace changes it. A matching hash does not prove schema, feature timing, or data quality.

**Build:** `load_h2(path) -> DataFrame`, `validate_source(frame) -> findings`, and a saved audit summary. Write a tiny synthetic CSV fixture for loader tests rather than committing H2.

**Checkpoint:** You can explain every column, the difference between source and derived data, and every cleaning decision made so far.

## 9. Define the prediction question and audit leakage

This chapter determines whether later scores mean what you claim.

### 9.1 Ask four questions per supervised task

1. At what exact point is the prediction made?
2. What outcome does it predict, over what horizon?
3. Which records belong to the population being predicted?
4. Which values are genuinely available at that point?

For ADR, the source article defines a transaction-derived average. Predicting that outcome is different from quoting an initial room price. For cancellation, predicting an eventual cancellation is different from identifying a booking already canceled.

### 9.2 The source snapshot limitation

The original paper describes reconstructing many inputs from booking/change-log records relative to the day before arrival. The exported CSV is not a complete booking-creation event history.

Consequences:

- Initial reservation fields may have been amended by the exported snapshot.
- DepositType is derived from payments before arrival/cancellation; its exported value is not necessarily an initial deposit policy.
- BookingChanges accumulates lifecycle information.
- Some bookings were canceled before the day-before-arrival point. Including them in a claim about still-future cancellations at that point is problematic.
- A clever time split cannot repair a feature observed after the claimed prediction time.

### 9.3 Defensible routes to discuss with your instructor

| Route | Claim | What you need to establish |
| --- | --- | --- |
| Booking-creation proxy benchmark | Estimate later outcomes using a conservative subset of reservation-like fields | Explicit assumptions that selected exported values approximate creation-time information; disclose amendments and unavailable histories |
| Operational snapshot study | Predict from a clearly described later snapshot/cohort | Establish which bookings were still at risk then and whether inputs/labels are available in the required order |
| Retrospective association study | Compare model relationships in this published snapshot | State that this is not validated prospective deployment; still justify predictors and exclude direct outcome proxies |

Do not assert that any route is automatically accepted by the course. Choose a prediction question consistent with the brief, explain the dataset limitations, and ask the TA when the source cannot support a strong operational claim.

An ADR-at-booking model may exclude many operational fields. A cancellation model may consider ADR only if its availability is defended; otherwise exclude it conservatively. Clustering should initially use booking attributes rather than ADR, IsCanceled, or final statuses; alternative segmentation definitions need explicit justification.

### 9.4 Build an allowlist, not only a blacklist

Create `docs/feature-policy.md` with rows:

~~~text
source column | task | meaning | available when? | include/exclude/quarantine | rationale
~~~

At minimum, exclude target/final-status fields from relevant inputs. Quarantine AssignedRoomType, BookingChanges, DaysInWaitingList, DepositType and mutable request fields until timing is justified.

An allowlist makes the model consume only approved columns. A blacklist alone can fail when an accidental new outcome-derived column appears.

Derived features inherit their inputs' availability. "Assigned room differs from reserved room" can still leak even after deleting AssignedRoomType itself. Aggregating the full dataset by country before splitting can also leak; learned aggregates belong inside fitting folds.

### 9.5 Three kinds of leakage

- **Target leakage:** the answer or a close consequence is in X.
- **Preprocessing leakage:** medians, categories, PCA directions, feature selection or other fitted transformations use holdout records.
- **Evaluation leakage:** repeated test inspection influences model, features, threshold, search space, or reporting choices.

Write an audit for regression, classification, and clustering separately. A high score is not proof of validity, and dropping two obvious status columns is not a complete audit.

**Checkpoint:** A teammate can challenge each chosen feature, and you can explain its source and timing without saying "it was in the CSV."

## 10. Freeze splits before modeling

### 10.1 Know the roles

| Partition | Allowed use |
| --- | --- |
| Training | Fit model and learned preprocessing; perform CV/search internally |
| Validation | Compare development choices and choose a classification operating threshold |
| Final test | Evaluate a frozen procedure after selection is finished |

Cross-validation repeatedly divides development data into fitting and scoring folds. It is not an extra independent final test. A transformation that sees every training row before CV can still leak between the CV folds.

### 10.2 Route A: stratified random benchmark

This is straightforward for an approximately exchangeable-booking question. It does not demonstrate future deployment.

**Example; use only after checking targets, duplicates, and the stated independence assumptions:**

~~~python
import numpy as np
from sklearn.model_selection import train_test_split

positions = np.arange(len(df))
development, test_positions = train_test_split(
    positions,
    test_size=0.15,
    random_state=42,
    stratify=df["IsCanceled"],
)
train_positions, validation_positions = train_test_split(
    development,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=df.iloc[development]["IsCanceled"],
)
~~~
This gives approximately 70% train, 15% validation, and 15% test; integer rounding affects exact counts. Stratification preserves class proportions approximately. It does not create independent customers or prevent identical feature patterns crossing partitions.

You may reuse these booking positions for regression to keep cohort bookkeeping consistent, but justify that choice for ADR. Never include IsCanceled in regression X merely because it was used to define stratification.

Persist the integer positions with the raw data hash, seed, strategy, and loader/version identity. Apply them with `.iloc` to the same row ordering. Assert pairwise-disjoint positions and complete coverage. Do not silently recreate the split whenever someone changes a seed.

### 10.3 Route B: grouped splitting

Use GroupShuffleSplit or a stratified group splitter when a defensible group must stay together. Keep the group labels aligned with X through every fold. Group-based fractions are not guaranteed to equal exact row-based fractions.

Without customer identifiers you cannot prove customer-disjoint splits. A hash of selected feature values can group identical input patterns, but it may combine distinct bookings. Document the meaning of that grouping; do not call it a customer ID.

If you group duplicates, define the group before splitting and use group-aware CV/search too. Fixing only the outer test split while ignoring groups inside CV does not solve the same issue.

### 10.4 Route C: chronological evaluation

Use whole date/cohort boundaries when the main claim concerns future bookings or arrivals. Reconstruct a full arrival date from year, English month, and day. If approximating booking-entry date as arrival minus LeadTime, disclose that exported arrival/amendment information limits that reconstruction.

A valid forward-looking fold also requires **training labels to have been available before the validation prediction time**. A booking created earlier but resolved later may need to be purged from the earlier training window. ReservationStatusDate can inform a separately documented availability audit; it is not a model feature.

TimeSeriesSplit orders folds by rows; it does not automatically enforce whole-day groups, label-resolution purging, a valid prediction timestamp, or duplicate-group isolation. Implement/test those rules explicitly if claiming them.

Chronological and random protocols can yield different results because they ask different questions. Declare the primary protocol before choosing models. If comparing both, report them as separate protocols, not interchangeable scores.

### 10.5 Selection and refitting

A beginner-friendly approach is to keep the selected pipeline fitted on the training partition, use validation for choices, and evaluate that pipeline once on final test data.

An alternative is to refit the frozen recipe on train plus validation before final testing. Freeze all choices first; decide how any probability calibration/threshold is obtained without using final test labels. Save and evaluate the actual refitted pipeline, not an earlier object with a different training history.

Do not repeatedly compare final-test scores to decide whether PCA, a new feature, or another model is better. If a genuine bug forces a correction after testing, document the correction and the fact that the holdout was already inspected.

**Build:** a split-manifest writer/reader and tests for disjointness, coverage, reproducibility, date/group constraints where claimed, and rejection of a changed source hash.

Source: [sklearn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).

## 11. Explore training data

You can inspect full-source schema/integrity to establish what file you received. Use training data for exploratory choices that affect features, preprocessing, outlier rules, or model selection. Treat validation as development evidence and keep the final supervised test sealed.

Create a notebook or `analyze` function that produces:

1. Shape, dtypes, missingness and cardinality tables.
2. Histograms/quantiles of ADR, LeadTime, nights and guest counts.
3. Cancellation prevalence and class counts.
4. Category frequencies for meals, segments, channels, countries and room codes.
5. Missingness patterns, not just totals.
6. Arrival-date coverage and seasonality.
7. Duplicate-pattern counts and their implications.
8. Target/input relationships on training data with appropriate caveats.
9. A table of unusual records and the rule used for each decision.

Use a log-scaled axis or a documented log transform to understand skew; do not automatically delete the right tail. Negative ADR, zeros and exceptional rates require investigation. If excluding records based on target range, you are changing the prediction population; report the rule and affected counts.

Distinguish **zero**, **unknown**, and **not applicable**. Company/Agent absence may have domain meaning; mean-imputing an identifier is meaningless.

Save plots intentionally:

~~~python
from pathlib import Path

import matplotlib.pyplot as plt

destination = Path("reports/figures")
destination.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(training_frame["LeadTime"].dropna(), bins=40)
ax.set(xlabel="Lead time (days)", ylabel="Bookings", title="Training-set lead time")
fig.tight_layout()
fig.savefig(destination / "training-lead-time.png", dpi=160)
plt.close(fig)
~~~
Here `training_frame` means the rows selected by your frozen training manifest. This example does not load a global dataset by itself.

For each figure write one observation, one possible explanation, and one limitation. Correlation does not establish feature availability, causation, or a justified cleaning rule.

**Checkpoint:** Another person can reconstruct each plot from a command and explain why no model-selection choice came from final test data.

## 12. Build preprocessing and features

### 12.1 Choose transformations by meaning and estimator

| Input type | Starting approach | Alternatives / concerns |
| --- | --- | --- |
| Numeric counts/durations | Training-fitted median imputation | Fixed domain value only if justified; missing indicators; training-only removal of unusable fields |
| Numeric for linear/distance models | StandardScaler after imputation | RobustScaler for outliers; log1p for nonnegative skew if justified |
| Nominal categories | Missing-value category plus one-hot encoding | Frequency grouping, hashing, or appropriate native categorical models |
| Agent/Company IDs | Categorical, or omit with rationale | Avoid arbitrary numeric distances; monitor cardinality |
| Calendar variables | Categorical or justified cyclic features | Month 12 and month 1 are adjacent; document representation |
| Binary fields | Consistent 0/1 representation | Validate allowed values and missing semantics |

StandardScaler learns a training mean and standard deviation. Trees usually do not need numeric scaling, but encoding and missing-value policy still matter. KNN, SVMs, PCA and distance-based clustering are especially sensitive to units.

One-hot encoding creates indicator columns. Ordinal encoding uses integer codes, which can impose an artificial ordering; it is not an automatic replacement for nominal one-hot encoding. Native categorical estimators have their own documented limits and missing/unseen-category rules.

### 12.2 A reusable preprocessor example

Write this yourself in `src/booking_ml/preprocessing.py` when you reach this stage:

~~~python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(numeric_columns, categorical_columns, dense=False):
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )
~~~
Explain every decision:

- ColumnTransformer selects declared columns rather than feeding the entire CSV to the estimator.
- Median imputation learns numeric medians during fitting.
- `keep_empty_features=True` keeps an all-missing training feature; numeric median imputation has a fallback for it. Decide whether such features should instead be removed using training-only evidence.
- The reserved missing category avoids assuming the most common category is correct. Check that your sentinel does not collide with a real category.
- `handle_unknown="ignore"` lets a previously unseen category transform to all-zero indicators for that feature. Decide whether an application should also warn about such inputs.
- `remainder="drop"` avoids passing undeclared columns through.
- Dense mode is explicit for estimators/representations requiring dense arrays.

Numeric scaling is included for teaching a shared linear/distance pipeline. You can make a tree-specific variant without scaling. Construct a **fresh** preprocessor/pipeline for each independent model experiment.

### 12.3 Memory is a design constraint

One-hot matrices can be mostly zeros. Sparse matrices store nonzero entries efficiently. Dense conversion can consume large amounts of RAM, especially when CV clones work concurrently.

Estimate dense bytes as `rows * transformed_columns * bytes_per_value`. At 79,330 rows and 1,000 float64 columns, the data alone is about 635 MB before copies, model memory and parallel workers.

Inspect the actual transformed shape, dtype and sparse/dense form on a small development run. Do not call `.toarray()` on a huge matrix just to fix an error.

Options: omit unjustified high-cardinality IDs, use training-fitted infrequent-category grouping, restrict a representation comparison to a justified input set, reduce parallelism, or choose an estimator suited to sparse input. Target encoding, if chosen, must be cross-fitted; computing target means globally leaks.

### 12.4 Feature engineering without hidden fitting

Candidate deterministic features include:

- TotalNights = StaysInWeekendNights + StaysInWeekNights.
- TotalGuests = Adults + Children + Babies.
- A complete arrival date or month sine/cosine pair, if the timing/interpretation is justified.

Preserve unknowns intentionally. If Children is missing, blindly treating it as zero changes meaning. A ratio involving zero total nights needs an explicit policy. A ratio involving ADR leaks into ADR regression.

A stateless function can copy a DataFrame, add approved features, and return it. Put reusable functions/classes in an importable module; avoid anonymous lambdas in serialized pipelines. Learned transformations belong inside pipeline fitting. If using a custom sklearn transformer, implement `fit(self, X, y=None)` returning self and `transform(self, X)` returning the transformed table; test that input data is not mutated.

Sources: [mixed-type pipelines](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html), [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

**Checkpoint:** You can explain which steps learn state, which preserve rows, what happens to unseen categories, and where fitting occurs during CV.


## 13. Your first complete learning example

Create a temporary learning script, for example `notebooks/toy_pipeline.py`, after you have written `make_preprocessor` from Chapter 12. This script uses synthetic values; it does not read H2 or establish a valid hotel feature policy.

~~~python
import numpy as np
import pandas as pd
from booking_ml.preprocessing import make_preprocessor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

toy = pd.DataFrame(
    {
        "LeadTime": [2, 6, 10, 20, 30, 50, 70, 90, 120, 150, 180, 210],
        "Adults": [1, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2],
        "MarketSegment": ["Direct", "Online", "Agency"] * 4,
        "ADR": [65, 85, 70, 90, 120, 95, 75, 100, 125, 105, 80, 110],
    }
)
X = toy.drop(columns=["ADR"])
y = toy["ADR"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)
pipeline = Pipeline(
    [
        (
            "preprocess",
            make_preprocessor(
                ["LeadTime", "Adults"],
                ["MarketSegment"],
            ),
        ),
        ("model", Ridge(alpha=1.0, solver="lsqr")),
    ]
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_valid)
assert predictions.shape == (len(X_valid),)
assert np.isfinite(predictions).all()
print("Toy validation MAE:", mean_absolute_error(y_valid, predictions))

unseen = X_valid.head(1).copy()
unseen["MarketSegment"] = "Unseen segment"
assert np.isfinite(pipeline.predict(unseen)).all()
~~~
Run it:

~~~powershell
python notebooks/toy_pipeline.py
~~~

Trace the operations:

1. The DataFrame holds inputs and a separate target.
2. The split creates development fitting/scoring rows.
3. `fit` learns numeric imputation/scaling and category vocabulary from X_train.
4. The transformed X_train trains Ridge against y_train.
5. `predict` transforms X_valid using the already fitted preprocessor.
6. MAE compares predictions with y_valid.
7. The extra prediction demonstrates the declared unknown-category behavior.

The fitted preprocessing is part of the model's input contract. Saving just Ridge would lose information needed to recreate its input matrix.

**Exercise:** Change alpha, record a second toy result, and explain why this is validation-based experimentation. Deliberately fit preprocessing on the full toy table and identify the methodological error even if the numeric score barely changes.

## 14. Regression experiments

First implement one task end to end before creating a registry of ten models.

### 14.1 Frame the target and baseline

Construct X using the approved regression allowlist and y from ADR. Keep them aligned with the split manifest. Do not include ADR, IsCanceled, final reservation status, or derived outcome proxies in X.

A `DummyRegressor(strategy="median")` predicts the training median and gives a useful MAE reference. Mean prediction is the corresponding squared-error reference. Dummy behavior should be fitted on training data too.

### 14.2 Choose at least three substantive families

A convenient all-sklearn route is Ridge, RandomForestRegressor and HistGradientBoostingRegressor. A route using your Colab GPU can substitute XGBoost for the boosting slot after compatibility/runtime checks.

| Model | What it learns | Main strengths | What to tune / watch |
| --- | --- | --- | --- |
| LinearRegression | Linear combination minimizing squared residuals | Transparent first comparison | Limited flexibility; encoding/collinearity; justify meaningful configurable variants if selecting this as a required tuned model |
| Ridge | Linear model with L2 coefficient penalty | Simple regularized baseline | alpha, preprocessing and representation |
| RandomForestRegressor | Average of randomized decision trees | Nonlinear interactions; no numeric scaling requirement | n_estimators, max_depth, min_samples_leaf, max_features; RAM and thread budget |
| HistGradientBoostingRegressor | Sequential trees correcting loss with histogram-based fitting | Flexible CPU boosting | learning_rate, max_iter, max_leaf_nodes, l2_regularization; dense-input route |
| XGBRegressor | Regularized gradient-boosted trees | Flexible CPU/GPU option | learning_rate, n_estimators, max_depth, subsample, colsample_bytree; explicit device policy |
| KNeighborsRegressor | Average of nearby training outcomes | Easy distance-based comparison | n_neighbors, weights, distance; scaling and slow prediction |
| SVR with RBF kernel | Regularized nonlinear function using similarities | Alternative nonlinear family | C, epsilon, gamma; scaling and substantial cost |

Prefer Ridge over an untunable-looking OLS setup if you want a straightforward, meaningful baseline/tuned pair. Do not invent a pointless search over an option that cannot affect the fitted predictions.

An untuned baseline still needs fixed random seeds and a declared execution budget. Settings such as a higher logistic-regression iteration limit to achieve convergence are not evidence of validation-driven tuning; disclose these implementation choices.

### 14.3 Metrics and their meaning

For residual `e_i = y_i - prediction_i`:

- **MAE** = average absolute residual. Its units match ADR.
- **MSE** = average squared residual. Large errors receive more weight; units are squared.
- **RMSE** = square root of MSE. It returns to ADR units while remaining sensitive to large errors.
- **R-squared** = 1 minus model squared error divided by the reference squared deviation from the evaluated set's mean. It can be negative. Constant-target sets need special care; do not interpret an undefined metric as excellent performance.

Example evaluator inputs are already aligned validation targets and predictions:

~~~python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

regression_metrics = {
    "mae": float(mean_absolute_error(y_valid, predictions)),
    "mse": float(mean_squared_error(y_valid, predictions)),
    "rmse": float(root_mean_squared_error(y_valid, predictions)),
    "r2": float(r2_score(y_valid, predictions)),
}
~~~
Choose a primary selection metric such as MAE before searching. Still report every rubric-required metric. A model can win MAE while losing RMSE; explain what that implies about the error distribution.

### 14.4 Manual implementation sequence

1. Write `build_regression_pipeline(config, feature_policy)`.
2. Start with one model and the correct preprocessing form.
3. Fit training rows, time the fit with `time.perf_counter()`, and predict validation rows.
4. Write metrics and configuration to a new run directory.
5. Save residual and predicted-versus-observed plots.
6. Repeat with the other families using the same declared comparison protocol.
7. Add tuning only after this basic route works.

For dense-only estimators, call a deliberate dense preprocessor, estimate memory, and log the representation. Do not silently drop rows/categories until it fits.

If trying `log1p(ADR)`, first check the target domain. Negative values may make the transform invalid. Use a target-transform wrapper such as TransformedTargetRegressor so inverse transformation is part of inference. Evaluate predictions in original ADR units. Log-transform modeling changes the loss behavior and needs its own experiment.

**Checkpoint:** You can explain what each regressor optimizes, why its preprocessing is appropriate, and why your chosen comparison metric matches the task.

## 15. Classification experiments

### 15.1 Target, imbalance, and substantive models

Use IsCanceled as a validated integer 0/1 target. State that class 1 is cancellation. Build X from the classification allowlist rather than automatically reusing the regression inputs.

Start with `DummyClassifier(strategy="prior")`. Accuracy alone can look acceptable when cancellations are rare; inspect class counts and use multiple metrics.

A useful three-family route is logistic regression, random forest, and an RBF-kernel SVC. This covers an ensemble and a kernel experiment without requiring a deep-learning framework.

| Model | Idea | Settings to investigate |
| --- | --- | --- |
| LogisticRegression | Linear score converted to class probability | C, class_weight, solver-compatible regularization; scale numerics and check convergence |
| RandomForestClassifier | Voting/averaging across randomized trees | depth, leaf size, trees, max_features, class_weight |
| SVC(kernel="rbf") | Nonlinear separating boundary from similarities | C, gamma; scale; bound sample size/runtime |
| GaussianNB | Class-conditional Gaussian features with independence assumptions | var_smoothing; consider whether representation matches assumptions |
| KNeighborsClassifier | Vote among nearby training records | neighbors, weighting, metric; scaling and runtime |
| Gradient boosting / XGBoost | Sequential trees for classification loss | search budget, depth, learning rate, class weighting/calibration choices |

A kernel SVM on tens of thousands of rows may be expensive. Establish a fixed, justified training subsample **within training data** if needed. For a fair three-family subset benchmark, train all compared families on that same subsample and use the same evaluation rows. Additional full-training-set runs for scalable models are useful but must be labeled as a different fitting budget.

### 15.2 Example construction

After declaring task-specific `numeric_columns` and `categorical_columns`:

~~~python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

classification_pipeline = Pipeline(
    [
        ("preprocess", make_preprocessor(numeric_columns, categorical_columns)),
        ("model", LogisticRegression(C=1.0, max_iter=2000)),
    ]
)
~~~
`max_iter` is an optimization/convergence limit. An unfinished optimizer is not a valid proof that the model family is weak. Inspect convergence warnings and document the response rather than suppressing them.

### 15.3 Labels versus scores versus probabilities

- A **label** is the final 0/1 decision.
- A **score** ranks examples, such as SVC's decision function.
- A **probability estimate** lies in [0,1], but is not automatically well calibrated.

ROC-AUC and average precision need scores/probabilities, not hard labels. Standard SVC can supply a decision function without expensive probability fitting.

For a fitted probability-capable pipeline and classification validation data:

~~~python
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

positive_column = list(classification_pipeline.classes_).index(1)
scores = classification_pipeline.predict_proba(X_valid)[:, positive_column]
threshold = 0.5  # Initial policy; choose any changed threshold on validation only.
labels = (scores >= threshold).astype(int)

classification_metrics = {
    "accuracy": float(accuracy_score(y_valid, labels)),
    "precision": float(precision_score(y_valid, labels, zero_division=0)),
    "recall": float(recall_score(y_valid, labels, zero_division=0)),
    "f1": float(f1_score(y_valid, labels, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_valid, scores)),
    "average_precision": float(average_precision_score(y_valid, scores)),
    "confusion_matrix": confusion_matrix(y_valid, labels, labels=[0, 1]).tolist(),
}
~~~
Here `classification_pipeline` must have been fitted on classification training data first. Do not run this using the regression example's y_valid.

For a binary SVC, inspect `classes_` and the meaning/sign of `decision_function`. Its default score threshold is 0, not probability threshold 0.5.

### 15.4 Interpret each metric

- Accuracy: fraction of all labels correct.
- Precision: among predicted cancellations, fraction actually canceled.
- Recall: among actual cancellations, fraction detected.
- F1: harmonic mean of precision and recall.
- Confusion matrix: with labels [0,1], rows are actual labels and columns predicted labels; identify TN, FP, FN and TP explicitly.
- ROC-AUC: ranking discrimination across thresholds; it is not accuracy or a cancellation probability.
- Average precision: a summary of the precision-recall curve. Label it as average precision; it is not identical to trapezoidal PR-curve area.

If a scoring set has only one class, some metrics are undefined. Diagnose the split, report the limitation, and store an explicit unavailable value rather than inventing 0 or 1.

### 15.5 Imbalance and threshold routes

Start with the original class distribution and evaluate it. Then compare:

1. Class-weighted models.
2. A changed validation-selected decision threshold.
3. Probability calibration where justified.
4. Optional resampling inside training folds.

Do not oversample the full dataset before splitting. If using imbalanced-learn, use its pipeline so sampling happens within each fitting fold. Synthetic interpolation across one-hot categories may create unrealistic records; do not assume SMOTE is automatically appropriate.

If tuning thresholds, decide the cost objective before looking for a convenient number. An F1 threshold, high-recall threshold and cost-minimizing threshold answer different questions. Save the threshold with the model metadata and use it consistently in the application.

Calibration is a separate question from ranking. Fit calibrators using valid training/development arrangements, never final test labels. Display uncalibrated output honestly rather than promising it is a precise real-world risk.

**Checkpoint:** Explain why a classifier can improve accuracy while becoming worse at identifying cancellations.

## 16. Cross-validation and hyperparameter tuning

### 16.1 What a fold does

For each candidate setting:

1. Start with an unfitted clone of the **whole pipeline**.
2. Fit preprocessing and estimator on that fold's fitting rows.
3. Transform/predict its scoring rows without refitting.
4. Compute the declared metric.
5. Aggregate fold results.

Do not precompute imputed/scaled/PCA-transformed data across all development rows before CV. Feature selection and any target-derived preprocessing also belong inside the fold.

### 16.2 Choose a search method

| Method | What you do | Good use | Limitation |
| --- | --- | --- | --- |
| Manual bounded search | Try a documented small list | Learning and initial debugging | Easy to drift or selectively report |
| GridSearchCV | Evaluate every listed combination | Small meaningful spaces | Product of choices grows rapidly |
| RandomizedSearchCV | Sample a fixed number of combinations/distributions | Clear compute budget | Coverage depends on declared distributions |
| Optuna | Define an objective and let a sampler propose trials | Conditional/large spaces | More setup; does not replace sound evaluation |

A model's hyperparameters differ from another model's. Build separate valid search spaces instead of handing one universal dictionary to every estimator.

### 16.3 Pipeline parameter names

A pipeline step called `model` exposes settings such as `model__C`.
A nested encoder can expose `preprocess__categorical__encode__min_frequency`.
Double underscores mean "descend into this named component."

**Classification tuning example**, after constructing the classification pipeline and classification-specific training partition:

~~~python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=classification_pipeline,
    param_distributions={
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    n_iter=8,
    scoring={"f1": "f1", "roc_auc": "roc_auc"},
    refit="f1",
    cv=cv,
    random_state=42,
    n_jobs=2,
    error_score="raise",
    return_train_score=True,
)
search.fit(X_train, y_train)
selected_pipeline = search.best_estimator_
print(search.best_params_)
print(search.best_score_)
~~~
This uses eight combinations times three folds, plus a final refit. `refit="f1"` chooses and refits using mean CV F1. `best_score_` is a development-CV score, not final-test evidence.

For regression, use a justified KFold/group/time protocol and a scorer such as `neg_mean_absolute_error`. sklearn makes losses negative so higher is consistently better; negate back for a positive MAE report.

For grouped data, pass the appropriate groups to a group-aware splitter/search. For temporal evaluation, use the audited chronological folds, including your label-availability and cohort rules.

### 16.4 Small search spaces to start from

These are initial teaching budgets, not empirically best settings:

- Ridge: alpha in [0.1, 1, 10, 100].
- Logistic regression: C in [0.01, 0.1, 1, 10], class_weight in [None, "balanced"].
- Forests: n_estimators in [100, 300], max_depth in [None, 10, 20], min_samples_leaf in [1, 5, 15].
- Histogram boosting: learning_rate in [0.03, 0.1], max_leaf_nodes in [15, 31], l2_regularization in [0, 1].
- RBF SVC: C in [0.1, 1, 10], gamma in ["scale", 0.01, 0.1].
- Clustering: use the separate unsupervised procedure in Chapter 18.

Do not evaluate every Cartesian product for every model just because you can. Profile one fit, estimate combinations times folds, reserve time for representation experiments and debugging, and log the planned budget before execution.

A convergence failure or invalid combination should be diagnosed. During development, `error_score="raise"` exposes the real error. In large searches you can record failed trials explicitly, but must not silently count failure scores as successful experiments.

### 16.5 Parallelism and early stopping

If CV uses two workers, setting every model to use all CPU cores can oversubscribe the runtime. Pick a single main parallelism level. On one GPU, concurrent CV workers can compete for the same memory; start GPU searches sequentially.

If using early stopping, ensure its monitoring data receives the correct training-fitted preprocessing. A raw external `eval_set` is not automatically transformed correctly merely because an estimator is wrapped in a Pipeline. Keep early-stopping data inside the valid development/fold procedure and never use the final test for it.

Log all candidates or the full `cv_results_`, best settings, metric, fold strategy, seed, fit time, and failure reasons. A reported best configuration without the search space/budget cannot be interpreted fairly.

**Checkpoint:** Calculate the fit count of your proposed search before running it. Explain precisely where a validation row is excluded from every learned preprocessing step.

Sources: [hyperparameter search](https://scikit-learn.org/stable/modules/grid_search.html), [metrics/scoring](https://scikit-learn.org/stable/modules/model_evaluation.html).


## 17. PCA and LDA

These are fitted representation choices, not generic decorations to add after seeing test scores.

### 17.1 PCA: directions of variance

PCA creates orthogonal linear combinations of prepared input features. Components are ordered by explained variance. It does not use y and does not know which directions are predictive.

You must:

1. Prepare an appropriate numerical representation with a documented scaling policy.
2. Fit PCA inside the training pipeline/folds.
3. Report explained-variance ratios and cumulative variance from a fitted development pipeline.
4. Justify the retained count.
5. Compare at least two supervised models with and without PCA under the same protocol.

A useful comparison set is Ridge(original/PCA) and logistic regression(original/PCA). These are representation variants of two models, not four independent model families.

**Example pipeline construction**, using the task's audited column lists:

~~~python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pca_classifier = Pipeline(
    [
        (
            "preprocess",
            make_preprocessor(
                numeric_columns,
                categorical_columns,
                dense=True,
            ),
        ),
        ("reduce", PCA(n_components=0.95, svd_solver="full")),
        ("model", LogisticRegression(C=1.0, max_iter=2000)),
    ]
)
~~~
Here 0.95 means retain enough components to explain at least 95% of variance in each fit, using the full-solver route. It is a starting hypothesis, not a universally best threshold. The fitted count can vary between CV folds. Do not fit PCA globally to choose a convenient number using final test data.

PCA on numeric-only features is another route: combine transformed numerical components with separately encoded categories and compare against the same original feature policy. A sparse representation may suit TruncatedSVD instead, but SVD is not interchangeable with centered PCA and does not remove the brief's explicit PCA requirement.

PCA can remove predictive low-variance information and can reduce interpretability. Plot validation performance and retained variance rather than assuming fewer dimensions always improves a model.

### 17.2 LDA: a supervised representation

LinearDiscriminantAnalysis can be a classifier itself or a supervised dimensionality-reduction step. For the required original/PCA/LDA comparison, use it as a transformation followed by the same chosen classifier.

For a binary task, the maximum number of discriminant components is **one**: generally at most min(number of input features, number of classes minus one).

~~~python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_classifier = Pipeline(
    [
        (
            "preprocess",
            make_preprocessor(
                numeric_columns,
                categorical_columns,
                dense=True,
            ),
        ),
        ("reduce", LinearDiscriminantAnalysis(n_components=1, solver="svd")),
        ("model", LogisticRegression(C=1.0, max_iter=2000)),
    ]
)
~~~
Pipeline fitting passes y into LDA during training. The default SVD-based route supports transformation; the `lsqr` solver is not a drop-in route for dimensionality-reduction transform. Inspect solver capabilities before changing it.

Fit LDA inside each fold because it uses labels. Keep class labels out of an unsupervised PCA/clustering transformation. Check finite numerical input, sufficient class samples, rank/conditioning, and memory.

### 17.3 Record the comparison

For each representation record: preprocessing, original transformed dimension, retained dimension, selection scores, training time, and memory considerations. If hyperparameters are selected separately for each representation, use comparable search budgets. Keep baseline and tuned model evidence; do not count representation variants as satisfying three different algorithms.

**Checkpoint:** Explain why PCA might discard a useful feature, why binary LDA produces at most one component, and why globally fitted LDA leaks labels into CV.

Sources: [PCA/decomposition](https://scikit-learn.org/stable/modules/decomposition.html), [LDA](https://scikit-learn.org/stable/modules/lda_qda.html).

## 18. Clustering

The aim is meaningful booking segments, not recovering IsCanceled with a disguised feature.

### 18.1 Choose inputs and geometry

Write a separate clustering allowlist. A numeric-only starting study might use justified stay/party/lead-time attributes. It must explain why omitted categories are not necessary for that segmentation question. A mixed-feature route can add properly encoded categories.

Before distance-based fitting:

- handle missingness according to a declared policy;
- scale numerical units;
- consider justified skew transformations;
- avoid double-weighting correlated totals and their component variables unintentionally;
- explain the implicit weighting of multiple one-hot columns versus numerical fields.

Categorical-only matching distances, Gower-style mixed distances, or k-prototypes are alternative specialized routes, with additional dependencies/memory tradeoffs. You do not need them if a justified numeric or encoded representation answers your question.

### 18.2 Choose three algorithms

| Algorithm | What it assumes/does | Optimize | Important limitation |
| --- | --- | --- | --- |
| K-Means | Assigns records to centroids to reduce squared within-cluster distances | k, initialization/restarts, representation | Favors compact roughly spherical groups; use full H2 where practical |
| GaussianMixture | Fits weighted Gaussian components with soft responsibilities | components, covariance_type, reg_covar, restarts | Dense low/moderate dimensions; distribution assumptions and convergence |
| AgglomerativeClustering | Repeatedly merges nearby groups | cluster count, linkage, valid metric combinations | Pairwise/hierarchical work can be expensive; justified sampling is allowed |
| DBSCAN, optional alternative | Density-connected groups and noise | eps, min_samples, representation | Noise handling and high-dimensional distances complicate comparison |

A convenient minimum is K-Means, GMM and sampled agglomerative clustering. Every selected algorithm needs both initial and optimized evidence.

Starting estimator examples:

~~~python
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
gmm = GaussianMixture(
    n_components=3,
    covariance_type="diag",
    n_init=3,
    random_state=42,
)
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward",
    metric="euclidean",
)
~~~
Three groups are a baseline hypothesis, not a conclusion. Ward linkage is tied to Euclidean geometry; not every metric/linkage combination is valid.

### 18.3 Full cohort versus a common benchmark sample

At 79,330 rows, a full dense pairwise float64 matrix would require about 50 GB before overhead. Actual algorithms may avoid that exact allocation, but quadratic-style work is still a serious reason to profile.

A transparent comparison plan:

1. Define an outcome-free analysis cohort and fit preprocessing according to that descriptive protocol.
2. Choose a fixed representative benchmark subset, for example 5,000 rows, and record its indices/seed.
3. Compare all three algorithms on that same benchmark representation/cohort.
4. Additionally fit K-Means on full H2 where practical, and GMM if feasible; label these full-cohort studies separately.
5. Do not claim a sampled hierarchical fit and a full-data fit have identical training budgets.
6. Use fixed, recorded scoring subsets for expensive internal metrics.

If clustering uses all H2 for descriptive analysis, keep it separate from supervised model selection. Do not inspect final supervised test outcomes through cluster profiles and then feed that knowledge into supervised feature/model decisions. If clusters become supervised features, fit their preprocessing and clustering within supervised training folds.

### 18.4 Internal scores

- **Silhouette:** compares within-cluster cohesion with separation; usually higher is better.
- **Davies-Bouldin:** compares cluster scatter and separation; lower is better.
- **K-Means inertia:** decreases as more clusters are added; it cannot alone justify arbitrarily large k.
- **GMM BIC/AIC:** likelihood-based complexity criteria; compare on the same data/representation and interpret assumptions.

For a numeric dense representation `Z` and matching labels, evaluate on fixed positions `score_positions`:

~~~python
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

Z_score = Z[score_positions]
labels_score = np.asarray(labels)[score_positions]
cluster_count = np.unique(labels_score).size

if not 2 <= cluster_count < len(labels_score):
    scores = {
        "silhouette": None,
        "davies_bouldin": None,
        "reason": "Internal scores undefined for this label arrangement",
    }
else:
    scores = {
        "silhouette": float(silhouette_score(Z_score, labels_score)),
        "davies_bouldin": float(davies_bouldin_score(Z_score, labels_score)),
    }
~~~
Use the feature space in which clustering was fitted for the principal internal scores. A prettier two-dimensional visualization is not a replacement for evaluating the actual geometry.

For DBSCAN, decide how to handle noise label -1. If excluding noise when scoring, report the excluded fraction and covered population. All-noise or single-cluster cases can make metrics undefined. A high score on a tiny retained subset does not establish better overall segmentation.

### 18.5 Systematic parameter optimization

Start with small predeclared spaces:

- K-Means: k in [2,3,4,5,6,8], fixed seed/restart policy.
- GMM: components in [2,3,4,5,6], covariance_type in ["diag","tied","full"], a small regularization list.
- Hierarchical: cluster count in [2,3,4,5,6], selected valid linkage/metric combinations.

Fit each candidate, record internal scores, cluster sizes, convergence status and timing. Use a declared primary internal metric plus stability/size/interpretation constraints. Consider multiple seeds or repeated representative subsets for stability; scores on one random sample are not guarantees.

Do not choose parameters by cancellation accuracy or an ADR target score. You may describe external outcomes after a segmentation is frozen, with clear separation from optimization and supervised holdouts.

### 18.6 Visualization and interpretation

Create a training/analysis-cohort PCA projection to two dimensions for plotting. If fitting in a higher-dimensional space, keep the original cluster labels and state that the projection hides information. Avoid concluding that overlapping plotted points prove the original clusters are useless.

For the dendrogram, use SciPy linkage/dendrogram on a documented manageable sample, such as 500-1,000 records. State its exact cohort and representation. A dendrogram from a different sample is illustrative, not the complete tree of the full H2 fit.

Profile groups in original human-readable units:

- size and share of bookings;
- median/IQR of lead time and nights;
- party-size patterns;
- category proportions where available;
- differences from overall cohort;
- caveats about sample and distance choices.

Cluster IDs are arbitrary labels. Cluster 0 in another seed need not represent the same segment. Use descriptive names only after examining profiles. Agglomerative clustering and DBSCAN do not offer the same natural new-row `predict` contract as K-Means/GMM; viewing saved analysis is sufficient for those exploratory methods under the brief.

**Checkpoint:** Explain each cluster in booking terms without using the outcome you were asked not to supply.

Sources: [clustering](https://scikit-learn.org/stable/modules/clustering.html), [mixtures](https://scikit-learn.org/stable/modules/mixture.html), [SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

## 19. Configs, versions, and run records

Add this layer after one complete model works. A config cannot train anything until your Python code reads it and constructs the corresponding operations.

### 19.1 Implement the flow by hand

~~~text
CLI/notebook chooses a config path
    -> load JSON and validate fields
    -> load matching data and split manifest
    -> resolve reviewed feature policy
    -> construct a fresh compatible preprocessing/model pipeline
    -> fit baseline or execute the declared search
    -> evaluate the declared development partition
    -> save evidence in a new run directory
~~~

Suggested functions to implement:

~~~python
def load_config(path): ...
def load_split_manifest(path, expected_data_hash): ...
def build_pipeline(config, feature_policy): ...
def train_run(config, data_path, artifact_root): ...
def compare_runs(run_paths): ...
~~~
These are interfaces for your implementation, not functions supplied by this guide.

### 19.2 Example recipe

Create `configs/regression/v1/ridge-baseline.json` when ready. Replace the capitalized policy placeholders with decisions you have actually reviewed:

~~~json
{
  "task": "regression",
  "version": "v1",
  "model": "ridge",
  "stage": "baseline",
  "representation": "original",
  "seed": 42,
  "split_id": "YOUR_FROZEN_SPLIT_ID",
  "feature_policy": "YOUR_REVIEWED_FEATURE_POLICY_ID",
  "prediction_point": "YOUR_DECLARED_PREDICTION_POINT",
  "selection_metric": "mae",
  "parameters": {"alpha": 1.0}
}
~~~

Use standard-library `json.load` or `json.loads` to read it into a dictionary. Validate:

- required/unknown keys, types, supported task/model/representation;
- a nonnegative valid integer seed, not a boolean;
- finite numeric parameters;
- task-compatible metric, representation and estimator settings;
- matching folder and version field;
- existing reviewed policies and split IDs;
- rejection of unresolved placeholders before a reported training run;
- no target/final-status predictors;
- safe output path components.

Use an explicit model registry or conditional factory, such as mapping "ridge" to a Ridge builder. Do not use `eval` on config values. Validate model parameters using the supported estimator interface and keep seed/thread policy consistent.

JSON is a good first route because it is dependency-free and strict. YAML is more convenient for comments but needs PyYAML and safe loading. Python config files are flexible but execute code and are less portable as declarative records. Hydra-style configuration is optional complexity, not a prerequisite.

### 19.3 Three identities, plus source history

| Identity | Example | Meaning |
| --- | --- | --- |
| Model/stage | ridge-baseline, ridge-tuned | Family and initial versus optimized procedure |
| Recipe version | regression/v1 versus regression/v2 | Meaningful feature/preprocessing/protocol change |
| Run ID | UTC timestamp plus random suffix | One execution, never overwritten |
| Source SHA | Git commit | Exact committed Python/config history |

Different tasks can advance versions independently. Multiple model families can share v1's protocol. Adding a tuned recipe alongside a baseline does not inherently require v2. Once results are used in reporting, preserve the recipe; add a new file/version for a changed experiment.

Keep the same evaluation partitions for directly comparable versions where possible. If the split, population or input availability changes, record a new protocol and do not describe the score difference as a pure model improvement.

### 19.4 Run contents and state

Create paths using `pathlib`, UTC timestamps, and a UUID suffix. Use `mkdir(..., exist_ok=False)` so an accidental collision cannot overwrite an old run.

~~~text
artifacts/regression/v1/<run-id>/
  config.json
  manifest.json
  metrics.json
  search-results.csv       # when a search happened
  feature-policy.json
  input-schema.json
  pipeline.joblib          # after successful fitting
  figures/
~~~

Record:

- task, model, stage, representation and recipe;
- source SHA and dirty state;
- data SHA-256, loader/normalization identity and row/cohort counts;
- split and feature-policy IDs;
- Python/platform and actual package versions;
- seed/search space/primary metric and fit counts;
- training/prediction/search times;
- selected parameters, validation evidence, warnings/failures;
- status: initialized, running, failed, completed.

A dirty flag alone does not preserve uncommitted source. For reported experiments, require a clean source checkout or preserve the exact patch/source snapshot. Do not claim that a SHA reconstructs edits it never contained.

Write to temporary files and replace finished files deliberately; mark a run completed only after its required outputs exist. On Colab, use local scratch storage during heavy I/O and persist completed outputs safely to Drive. Check file hashes when transferring important artifacts. An interrupted partial file is not a valid model.

Keep bulk artifacts ignored. Promote reviewed comparison tables/figures to `reports/` with run references when these outputs exist.

**Checkpoint:** A teammate can identify exactly which code, data, split, recipe, environment and model produced one result.

## 20. Colab and your compute budget

Your 200 compute units belong to your Google account. Teammates running under their accounts use their own runtime/resources. Shared code and shared notebook access do not share the executing machine.

A practical team arrangement: members implement and commit their own code/configs; everyone checks small runs; you execute heavier agreed experiments under your account from exact commits; share reviewed results and artifact references.

### 20.1 Write your own thin launcher

Create `notebooks/colab_runner.ipynb` in the new repository. It should call your package rather than contain another training implementation.

**Cell 1: choose the source revision.** Replace the repository URL:

~~~python
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/YOUR_USERNAME/hotel-booking-lab.git"
REVISION = "main"  # Use an exact commit SHA for reported experiments.
WORKDIR = Path("/content/hotel-booking-lab")

if not WORKDIR.exists():
    subprocess.run(["git", "clone", REPOSITORY, str(WORKDIR)], check=True)
else:
    origin = subprocess.check_output(
        ["git", "-C", str(WORKDIR), "remote", "get-url", "origin"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(WORKDIR), "status", "--porcelain"],
        text=True,
    ).strip()
    if origin != REPOSITORY or dirty:
        raise RuntimeError("Review the existing checkout before switching revisions.")

subprocess.run(["git", "-C", str(WORKDIR), "fetch", "origin"], check=True)
chosen = "origin/main" if REVISION == "main" else REVISION
subprocess.run(["git", "-C", str(WORKDIR), "checkout", "--detach", chosen], check=True)
os.chdir(WORKDIR)
print(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
~~~
`--detach` selects an exact code state without pretending to edit the shared main branch. To try a remote feature branch, use a reference such as `origin/feat/regression-baseline`. For published comparisons, prefer its exact commit SHA.

**Cell 2: install the new project's package.**

~~~python
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "."],
    check=True,
)
subprocess.run([sys.executable, "-m", "booking_ml", "doctor"], check=True)
~~~
This installs core dependencies from the pyproject you wrote, not code from the old project. Add chosen extras/packages only when needed. If installing/upgrading a package already imported into the kernel, restart the runtime as required and rerun setup. Do not force incompatible Python versions with ignore flags; choose a supported runtime/package arrangement.

**Cell 3: mount your Drive and copy the supplied data.**

~~~python
import shutil

from google.colab import drive

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/hotel-booking-lab")
data_path = WORKDIR / "data" / "raw" / "H2.csv"
source_data = DRIVE_ROOT / "data" / "H2.csv"
if not source_data.is_file():
    raise FileNotFoundError(f"Provide H2.csv at {source_data}")
data_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source_data, data_path)
artifact_root = DRIVE_ROOT / "artifacts"
artifact_root.mkdir(parents=True, exist_ok=True)
~~~
Create the corresponding folder and put your dataset in Drive before this cell. Mounting grants the notebook access to your Drive; review the notebook you execute. Copying to local runtime storage avoids repeated slow data reads from mounted Drive.

**Cell 4: execute your implemented interface.** This is a proposed invocation until Chapter 22's `train` command exists:

~~~python
# Proposed interface: implement train and these arguments before running.
subprocess.run(
    [
        sys.executable,
        "-m",
        "booking_ml",
        "train",
        "--config",
        "configs/regression/v1/ridge-baseline.json",
        "--data",
        str(data_path),
        "--output-root",
        str(artifact_root),
    ],
    check=True,
)
~~~
Passing paths explicitly is easiest to understand. If later using environment variables, implement their resolution yourself; setting `HOTEL_ML_ARTIFACT_ROOT` does not automatically redirect a script that ignores it.

For a private new repository, this public-clone snippet needs explicit approved authentication. Opening a private notebook in the browser does not automatically authenticate a git subprocess. Another route is uploading your own clean source archive and installing it locally in the runtime, while recording the source SHA that produced the archive. Never embed a reusable token in notebook cells.

### 20.2 CPU versus GPU

Standard pandas/NumPy/sklearn CPU pipelines do not become GPU pipelines just by selecting a GPU runtime. Forests, kernel SVC and the PCA examples here use their normal CPU implementations.

For an optional XGBoost GPU route:

~~~powershell
python -m pip install xgboost
~~~

The model construction inside a compatible GPU runtime can use:

~~~python
from xgboost import XGBRegressor

gpu_regressor = XGBRegressor(
    tree_method="hist",
    device="cuda",
    n_estimators=200,
    max_depth=6,
    random_state=42,
)
~~~
Check the installed XGBoost version's CUDA/device requirements against Colab's actual runtime. Verify a small fit uses the intended device, inspect warnings/configuration, and measure time. A missing/unsupported GPU may error or fall back; do not report GPU use based solely on the constructor text. You do not need a GPU driver installation on your local Windows machine merely to execute GPU work in Colab.

### 20.3 Spend compute deliberately

1. Inspect the runtime's current compute-unit rate, memory, and balance in the Colab UI.
2. Run one small smoke fit; record elapsed time and memory behavior.
3. Estimate search candidates times folds, plus refits and representation variants.
4. Start with small data/budgets for code debugging.
5. Use the full declared cohort for final comparable runs where practical.
6. Run searches with controlled parallelism; one GPU is not several independent GPUs.
7. Save completed run evidence and checkpoints to persistent storage.
8. Disconnect/delete the runtime when finished, after checking outputs are preserved.

Two hundred compute units are not automatically two hundred GPU hours. Rates, hardware and availability vary. A paid balance also does not promise unlimited uninterrupted execution. Do not assume closing a tab preserves a long job under every plan.

Sharing a notebook does not share Python variables or installed libraries with teammates. Clear saved cell outputs and execution counts before committing notebooks. Saving a notebook also does not automatically commit the package files edited under /content. Prefer changing code in your local clone, pushing a commit, and running that revision remotely.

**Checkpoint:** Reconnect to a fresh runtime and reproduce one small run using only the repository, your data, and declared persistent outputs.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [XGBoost GPU guide](https://xgboost.readthedocs.io/en/stable/gpu/index.html).


## 21. Save pipelines and implement inference

Training and inference have different jobs. Training learns state from labeled development data. Inference applies saved state to a new booking without knowing its outcome.

### 21.1 Persist the complete transformation chain

For a fitted development pipeline and an existing run directory:

~~~python
from pathlib import Path

import joblib
import numpy as np

run_dir = Path("artifacts/toy/serialization-check")
run_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, run_dir / "pipeline.joblib")
restored = joblib.load(run_dir / "pipeline.joblib")

before = pipeline.predict(X_valid)
after = restored.predict(X_valid)
np.testing.assert_allclose(before, after)
~~~
Here `pipeline` and `X_valid` can come from the synthetic regression example. The path is for that toy persistence check; real runs need their own unique IDs.

Only load artifacts produced by trusted code/people. Joblib/pickle loading can execute code. Do not accept arbitrary uploaded model files in a public app. Record and check compatible Python/library versions; cross-version loading is not a reproducibility guarantee.

If you used custom functions/transformers, they must live in importable package modules available when loading. A function defined only in a temporary notebook cell is a fragile deployment dependency.

### 21.2 Save the input contract and decision policy

Alongside the fitted pipeline, save:

- approved raw column names and semantic types;
- required fields and whether an explicit missing value is accepted;
- numeric range policies and category/unseen-category behavior;
- source-name mapping;
- task and prediction point;
- feature-policy and split IDs;
- classification positive class, threshold, score type, and calibration information;
- model selection criterion and relevant evaluation summary;
- training data/source/environment identity.

A regressor's output is a numeric ADR estimate. A classifier may return a label and probability/decision score with the saved threshold. Do not label an SVC decision function as a probability.

Choose an explicit selected-model manifest or argument. "Load the latest file in the folder" is not a scientific model-selection rule.

### 21.3 Write one inference function used everywhere

Implement `predict_bookings(artifact_path, booking_records)`:

1. Resolve the chosen trusted artifact bundle.
2. Load/check its metadata and pipeline.
3. Construct a DataFrame using the expected input schema.
4. Detect absent required fields, invalid numeric strings, invalid categories/ranges and extra outcome fields.
5. Apply documented missing/unseen policies.
6. Call `predict` and the appropriate score method.
7. Apply the saved threshold/class mapping if classification.
8. Return structured results plus model/run identity.

A missing column is not the same as a known column containing a missing cell. Reject incomplete schemas unless you deliberately support a documented default. Do not silently fill every omitted booking field with zero.

Stateless engineered features should be included in the saved pipeline or in a stable, tested shared input path used both during training and inference. Never recalculate a different feature definition in the UI.

### 21.4 Test the actual demonstration environment

If training used GPU-capable packages, verify saved models can load and predict on the machine you will demonstrate. Use supported CPU inference settings if needed; do not assume the training hardware configuration transfers automatically.

Download/preserve complete fitted pipelines locally before the discussion. A GitHub checkout alone will not include ignored fitted binaries. Provide a clear message if an artifact is absent; prediction must not secretly retrain or return hard-coded output.

**Checkpoint:** A fresh Python process can load an artifact and reproduce saved-input predictions without loading training data or calling fit.

Source: [model persistence](https://scikit-learn.org/stable/model_persistence.html).

## 22. Build the application

The CLI you started in Chapter 6 should grow by dispatching to package functions. Do not place all scientific logic inside argument parsing or UI callbacks.

### 22.1 Proposed command design

Implement these progressively; only `doctor` exists after the early package exercise:

| Proposed command | Behavior |
| --- | --- |
| `doctor` | Show runtime, package versions, paths and artifact availability |
| `analyze --data ...` | Validate source and produce/view appropriate data analysis |
| `train --config ... --data ... --output-root ...` | Run a baseline recipe and persist evidence |
| `tune --config ... --data ...` | Execute the declared search and save its results |
| `compare --task ...` | Display saved baseline/tuned comparisons and search summaries |
| `cluster --config ... --data ...` | Run or display the selected unsupervised analysis |
| `predict --task ... --artifact ... --input ...` | Load a saved pipeline and predict JSON/CSV booking inputs |
| `evaluate-final --selection ...` | Deliberately evaluate the frozen selected procedure on sealed test data |

The brief's command names are examples, not mandatory spelling. What matters is that users can reach every required capability without opening source code/notebooks.

Use argparse subparsers, `Path` arguments, declared choices, readable help and exit codes. A command handler should validate its arguments then call a function, such as `handle_predict(args)` calling `predict_bookings(...)`.

Use logging for progress and diagnostics; write useful errors to stderr and return nonzero exit status. Avoid hiding all exceptions in an empty catch block. During development, make the underlying error visible.

### 22.2 CLI route

For a first complete submission, implement:

- a dataset summary and paths to saved analysis;
- readable comparison tables, chosen model and selection rationale;
- training and/or explicit artifact-loading actions;
- access to tuning search/configuration results;
- regression and classification prediction input;
- clustering scores, profiles and saved plots;
- relevant validation/final-evaluation information.

JSON input is a useful first route because it supports reproducible demonstrations and batches. Generate an example input file from the selected artifact's actual raw schema, not a guessed three-field object for a twenty-field model.

**Exercise:** Write `examples/booking.json` only when a model schema exists. Run the same input through your inference function and CLI and assert matching outputs.

### 22.3 Graphical routes

| Route | Additional installation | Choose it when | Main work to understand |
| --- | --- | --- | --- |
| Streamlit | `python -m pip install streamlit` | You want a Python dashboard quickly | Reruns, widgets, state, caching and explicit actions |
| Gradio | `python -m pip install gradio` | You prefer a compact interaction interface | Compose all required views, not only a prediction button |
| FastAPI + Uvicorn | `python -m pip install fastapi uvicorn` | You want a separate HTTP backend | Request validation, endpoints, process lifecycle, frontend integration |
| Flask | `python -m pip install flask` | You want a lightweight web service/template app | Input handling, routes and application structure |
| Tkinter | Python/Tk availability depends on platform | You prefer a local desktop interface | Event loop and not blocking the UI during training |
| PyQt | Chosen Qt Python binding | A richer desktop app is worth the effort | Widgets, signals and background work |
| React plus API | Node/package manager plus backend | Your team deliberately wants frontend engineering | Another language/toolchain, HTTP contracts, CORS and deployment |

CLI plus Streamlit is usually a manageable learning progression for this project. A React frontend does not improve the scientific evaluation by itself.

For Streamlit, create `apps/app.py` only at this stage and launch:

~~~powershell
python -m streamlit run apps/app.py
~~~

Learn the [official app tutorial](https://docs.streamlit.io/get-started/tutorials/create-an-app). Then implement tabs/views for dataset analysis, experiments/comparisons, predictions, and clustering.

Important behaviors:

- Loading a pipeline is allowed during app startup; retraining every time a widget changes is not.
- Put training behind an explicit action and clearly show progress/run identity.
- Share inference functions with the CLI.
- If caching resources, key cache invalidation on artifact identity/version; do not keep a stale changed file under the same cache key.
- Treat fitted estimators as prediction resources, not shared mutable objects to refit unpredictably.
- Validate inputs using the stored schema rather than maintaining a contradictory second form definition.
- Display model identity and metric context, not an unexplained number.

If you expose an HTTP API, keep prediction schemas and versioning explicit. Authentication, hosting and production monitoring are optional expansion work, not substitutes for the required local demonstration.

**Checkpoint:** A teammate who has never opened the source can launch the app, inspect comparisons, and obtain genuine saved-model predictions.

## 23. Test and automate checks

Testing should catch meaningful failures. A successful script run is necessary but does not establish correct leakage handling or scientific assumptions.

### 23.1 Tests to implement

| Area | Useful evidence |
| --- | --- |
| Loading | Missing file, wrong/missing/extra source columns, explicit missing markers |
| Splits | Disjointness, coverage, stable IDs, changed-hash rejection, claimed group/time rules |
| Feature policy | Targets/status fields excluded; derived features respect approved inputs |
| Preprocessing | Training-only learned statistics; stable column order; unseen categories; row count preserved |
| Models | Tiny synthetic fit/predict returns valid shapes/finite outputs |
| Classification | Correct positive-class mapping and persisted threshold |
| Clustering | Undefined score cases and DBSCAN noise policy handled honestly |
| Persistence | Predictions match before/after reload in a fresh process |
| CLI | Help works; invalid command/input returns a useful nonzero error |
| Application | Missing artifacts give a clear message; prediction does not call fit |
| Reproduction | A clean clone can follow README commands without hidden local files |

### 23.2 A concrete preprocessing test

After writing `make_preprocessor`, create `tests/test_preprocessing.py`:

~~~python
import unittest

import numpy as np
import pandas as pd
from booking_ml.preprocessing import make_preprocessor


class PreprocessingTests(unittest.TestCase):
    def test_holdout_transform_does_not_refit_training_statistics(self):
        train = pd.DataFrame(
            {
                "LeadTime": [0.0, 10.0, np.nan],
                "MarketSegment": ["Direct", "Agency", "Direct"],
            }
        )
        holdout = pd.DataFrame(
            {
                "LeadTime": [100000.0],
                "MarketSegment": ["Previously unseen"],
            }
        )
        preprocessor = make_preprocessor(["LeadTime"], ["MarketSegment"])
        preprocessor.fit(train)
        imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
        before = imputer.statistics_.copy()
        transformed = preprocessor.transform(holdout)
        np.testing.assert_allclose(before, [5.0])
        np.testing.assert_array_equal(before, imputer.statistics_)
        self.assertEqual(transformed.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
~~~
This checks that transforming a very different holdout does not relearn the median and that an unknown category is handled. It does not prove the source variables were observed at the right time; that still needs domain review.

Run:

~~~powershell
python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
~~~

Apply fixes intentionally:

~~~powershell
python -m ruff format .
python -m ruff check --select I --fix .
~~~

Formatting and import sorting are separate operations. Read other lint diagnostics rather than applying every possible fix blindly. `unittest` is built in. If choosing pytest instead, install it and use equivalent tests/fixtures consistently.

### 23.3 Continuous integration

Create `.github/workflows/ci.yml` when the basic tests run locally:

~~~yaml
name: Checks
on:
  push:
  pull_request:
permissions:
  contents: read
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
      - run: python -m pip install -e ".[dev]"
      - run: python -m ruff check .
      - run: python -m ruff format --check .
      - run: python -m unittest discover -s tests -v
      - run: python -m booking_ml doctor
~~~

The official action READMEs showed v7 usage when checked for this guide. Check their release/runner requirements when you implement it. For stronger dependency control you can pin actions to reviewed commit SHAs and update them deliberately.

The runner checks out committed code, installs an environment, and executes your checks. Do not put H2 or a long tuning job into ordinary CI. Synthetic fixtures should make checks fast and independent of private data/GPU access.

Later add a Windows/Python matrix if useful, and validate declared minimum/selected runtime support. This sample is a future workflow for your new repository; it has not been executed there by this guide.

**Checkpoint:** Deliberately break a tiny test on a temporary branch, see CI fail, fix it, and see it pass. Do not merge the deliberately broken state.

Sources: [checkout action](https://github.com/actions/checkout), [setup-python action](https://github.com/actions/setup-python).

## 24. Compare, select, and evaluate once

Create reviewed comparison tables from run records rather than manually typing convenient numbers into a report.

### 24.1 What one row should contain

~~~text
task | model | version | baseline/tuned | representation | data/cohort hash
split ID | training size | CV/validation metric(s) | fit/search time
best settings | seed | source SHA | run ID | notes
~~~

Separate tables/protocol labels should distinguish:

- full versus sampled training cohorts;
- random versus chronological evaluation;
- different feature availability assumptions;
- different preprocessing/representations;
- validation selection evidence versus final-test evaluation.

A better score with a different population or relaxed leakage policy is not a pure improvement.

### 24.2 Select on development evidence

Declare primary metrics and any constraints before choosing:

- Regression: perhaps lowest validation/CV MAE with acceptable runtime.
- Classification: perhaps best validation F1, or highest recall while meeting a minimum precision.
- Clustering: internal quality plus meaningful sizes/stability/interpretation, without target labels.
- Application: prediction latency, artifact size, robustness and explanation may break development-score ties.

Use fold means and variability, not just a favorable single split. Standard deviation across CV folds is descriptive variability, not automatically an independent 95% confidence interval. If using bootstrap intervals, resample at the appropriate unit/group; naive independent-row bootstrap is unsuitable for strongly dependent bookings.

Do not run final tests on every candidate throughout development and then pick the highest final-test score. For the required final comparison, freeze the selected candidates/procedure first, perform the agreed final reporting, and do not use those scores for another round of model selection.

### 24.3 Final evaluation ceremony

1. Freeze feature policy, preprocessing, representation, hyperparameters, calibration/threshold, and selected model IDs.
2. Commit the source/configs and record the SHA.
3. Follow the declared train-only or final development-refit policy.
4. Load the sealed test indices and verify the raw source hash.
5. Predict without fitting any component on test rows.
6. Write all required metrics and plots with explicit `final_test` labels.
7. Preserve the fitted artifact and selection/evaluation manifests.
8. Explain failures and limitations as carefully as successes.

Diagnostics after final testing can explain errors, but do not use them silently to improve the chosen model and reuse the same holdout as fresh evidence. Model inspection/permutation importance can be done on development validation data while choices are still being made.

**Checkpoint:** You can name the exact evidence that chose the deployed model and demonstrate that the final test did not participate in that choice.


## 25. Report, package, and demonstrate

The report explains the system and scientific decisions; it is not a pasted notebook output.

### 25.1 Report structure

1. **Problem and dataset:** task definitions, source, H2 validation and population.
2. **Prediction points and feature policy:** availability assumptions and limitations.
3. **EDA and preparation:** missingness, duplicates, unusual values, transformations and their justification.
4. **Experimental protocol:** data/split identity, groups/time considerations, primary metrics, seeds and environment.
5. **Regression:** every substantive baseline/tuned model and validation/final-test evidence.
6. **Classification:** every baseline/tuned model, confusion matrices, ranking metrics and threshold rationale.
7. **PCA/LDA:** variance, retained dimensions, original-versus-reduced comparisons and limitations.
8. **Clustering:** inputs, three initial/optimized algorithms, internal scores, sampling, plots, dendrogram and segment meaning.
9. **Optimization:** search spaces, budgets, selection criteria, failures and best settings.
10. **Leakage audit:** direct and indirect risks across all three tasks and the complete workflow.
11. **Model selection and application:** saved artifacts, interface, example genuine predictions and runtime behavior.
12. **Reproduction, contributions and limitations:** commands, source/data IDs, member responsibilities, residual uncertainty and references.

Every table needs its evaluation partition/protocol, metric direction, and run references. Every plot needs readable axes, units, cohort and a caption. Include inconvenient outcomes and failed hypotheses when they explain the selected approach.

### 25.2 Final README checklist

Your new repository's README should tell a stranger:

- what the application does and what is actually implemented;
- how to clone, create an environment and install declared dependencies;
- where to put the user-supplied H2.csv;
- exact commands to analyze, reproduce experiments, load artifacts, predict and launch the interface;
- how Colab and persistent outputs work;
- how configs/versions/runs map to code and data;
- which reports/comparisons to read;
- team identities and real contribution process;
- the repository URL and submitted source SHA.

Commands must be tested from a clean clone. If an optional GPU or UI dependency is necessary for a particular route, say exactly when/how to install it.

### 25.3 Release and ZIP procedure

Before each milestone:

1. Check working tree and tests.
2. Verify that no raw dataset, credentials, environment, or fitted pipeline is tracked.
3. Commit the exact source/configs/report/results being submitted.
4. Record `git rev-parse HEAD`.
5. Build a source archive from committed files.
6. Put that source SHA and GitHub URL into the README **inside the submission ZIP**.
7. Inspect the ZIP contents and reproduce its documented commands.
8. Tag the submitted source version and retain the submitted archive.

There is a small practical trap: adding a commit's own SHA to a tracked README changes that file and creates a different commit. Resolve this by packaging from a fixed source commit, then adding its SHA to the exported README during packaging. State that this is release metadata; do not invent a self-referential commit.

A packaging example you can implement as `scripts/package_submission.py` **and commit before running**:

~~~python
import json
import subprocess
import zipfile
from pathlib import Path

if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
    raise RuntimeError("Commit or deliberately resolve changes before packaging.")

source_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()
destination = Path("dist")
destination.mkdir(exist_ok=True)
raw_zip = destination / f"source-{source_sha[:12]}.zip"
final_zip = destination / f"milestone1-{source_sha[:12]}.zip"
if raw_zip.exists() or final_zip.exists():
    raise FileExistsError("Archive exists; inspect it rather than overwriting it.")

subprocess.run(
    [
        "git",
        "archive",
        "--format=zip",
        f"--output={raw_zip}",
        "HEAD",
    ],
    check=True,
)
metadata = {
    "source_commit": source_sha,
    "repository": "https://github.com/YOUR_USERNAME/hotel-booking-lab",
    "milestone": "milestone1",
}
with zipfile.ZipFile(raw_zip) as source_archive:
    members = source_archive.infolist()
    if not any(member.filename == "README.md" for member in members):
        raise RuntimeError("Archive needs README.md.")
    for member in members:
        name = member.filename
        raw_data = (
            not member.is_dir() and name.startswith("data/raw/") and name != "data/raw/.gitkeep"
        )
        model_file = Path(name).suffix in {".joblib", ".pkl", ".pickle", ".onnx", ".pt", ".h5"}
        if raw_data or name.startswith(("artifacts/", ".venv/")) or model_file:
            raise RuntimeError(f"Unexpected submission content: {name}")
    with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for member in members:
            content = source_archive.read(member.filename)
            if member.filename == "README.md":
                content += (
                    f"\n\nSubmission source commit: {source_sha}\n"
                    f"Repository: {metadata['repository']}\n"
                ).encode("utf-8")
            output.writestr(member, content)
        output.writestr("SUBMISSION.json", json.dumps(metadata, indent=2))
print(final_zip)
~~~
Replace the repository placeholder. Change the milestone label deliberately for the final submission. Because the archive uses committed files, an uncommitted report is not included. `.gitignore` does not protect a binary that was already tracked; inspect `git ls-files` before packaging.

This method preserves tracked dotfiles rather than relying on file-selection tools that may omit hidden entries. It does not constitute a secret scanner: inspect the actual archive and tracked content too. No directory tree is deleted.

Tag the fixed source after verification:

~~~powershell
git tag -a milestone-1 -m "Milestone 1 submission"
git push origin main
git push origin milestone-1
~~~

A tag is a release pointer, not another experiment recipe version. Do not move a published submission tag to a different commit. Keep fitted model artifacts separately available locally for the discussion; the source ZIP excludes them as required.

### 25.4 Live demonstration rehearsal

From a fresh terminal on the actual presentation machine:

1. Confirm the intended interpreter and dependencies.
2. Verify data/model files exist locally; do not depend on a last-minute cloud session.
3. Start the application.
4. Display dataset summary and reviewed model comparisons.
5. Load the selected regression and classification pipelines.
6. Predict several valid new inputs and explain their fields.
7. Show a useful error for a deliberately invalid input.
8. Inspect clustering results, parameter selection, PCA/LDA comparisons and search evidence.
9. Explain which results are validation and which are final test.
10. Ask each member to explain a component they did not personally implement.

Practice artifact reload after closing/reopening the app. Demonstrate genuine model inference; never hard-code a plausible answer for the presentation.

## 26. Debugging reference

Start with the smallest failing example. Read the full error, identify the failing layer, change one thing, rerun the relevant check, and record the fix. Do not launch another long Colab search to debug a CSV spelling error.

| Symptom | Likely question | First useful action |
| --- | --- | --- |
| `python` or `py` not found | Is Python installed and terminal PATH refreshed? | Use the official launcher/install instructions; open a new terminal |
| Activation blocked | Is PowerShell refusing scripts? | Use the explicit .venv Python path; no policy bypass is needed |
| Module not found after installation | Did pip install into another interpreter? | Print sys.executable; run that interpreter's `-m pip show`; verify editable install |
| Notebook has different package versions | Wrong kernel or already-imported modules? | Print sys.executable in the cell; select/restart the intended kernel |
| H2.csv missing | Wrong working directory or Drive path? | Print Path.cwd() and the resolved source path |
| Wrong schema/extra index column | Different mirror/export? | Inspect raw header and documented mapping; do not silently drop unknown columns |
| String cannot convert to float | Categorical input bypassed encoding? | Inspect selected columns, dtype policy and actual pipeline |
| NaN/infinity rejected | Missingness or invalid derived arithmetic? | Inspect input and transformed matrix; check division by zero/domain errors |
| Unknown category rejected | Encoder policy missing? | Choose/document unknown handling; refit only training preprocessing |
| CSR/dense-input error | Estimator/representation incompatible? | Inspect shape/type and estimate dense memory before changing representation |
| Memory crash in CV/PCA | Too many columns/copies/workers? | Reduce parallelism, justify representation/cardinality, profile one fit |
| Convergence warning | Optimization unfinished or poorly conditioned? | Check scaling, model solver/limits and invalid data; do not just suppress it |
| Implausibly perfect cancellation results | Outcome proxy or duplicate/split leakage? | Audit all inputs/derived features, source timing and folds |
| Negative R-squared | Worse than evaluated-set mean reference? | Compare dummy baseline, errors and split; negative is possible |
| Precision/F1 warning | No positives predicted or absent class? | Inspect confusion matrix/class counts; use explicit undefined-metric policy |
| ROC-AUC undefined | Evaluation set has one class? | Investigate split/group/time coverage rather than inventing a score |
| GPU connected but idle | Is the estimator actually GPU-enabled? | Inspect chosen algorithm/device and warnings; use CPU for CPU work |
| CUDA error | Runtime/wheel/hardware mismatch? | Check installed XGBoost/runtime requirements; use supported configuration |
| Colab disconnect lost files | Outputs only in temporary storage? | Restore saved code/data from Git/Drive and resume from valid completed records |
| Pipeline cannot load | Missing module or incompatible environment? | Install the package/recorded environment; use trusted artifacts only |
| Notebook works only after arbitrary cells | Hidden state/order dependence? | Restart and run top to bottom; move reusable logic into modules |
| Git push rejected | Behind remote or wrong access? | Check branch/status/remotes and collaborator authentication; fetch/review before merging |
| Git reports dubious ownership | Repository owned by another OS identity? | Verify ownership and the specific trusted path; do not trust every directory globally |
| Secret/data accidentally staged | Did ignore rules apply to untracked files only? | Inspect staged diff, unstage affected files, fix ignore policy before commit |
| CLI says command is invalid | Was the proposed command actually implemented? | Inspect `--help`; the guide's proposed interfaces are not preinstalled commands |

For a scientific discrepancy, check source hash, row ordering, split ID, feature policy, seed, environment, fit budget and selection metric before blaming the algorithm.

## 27. Completion and understanding checklists

### 27.1 Minimal mathematics you should be able to explain

You do not need to derive every library implementation to build the system, but you should understand its central objective.

| Concept | Essential idea |
| --- | --- |
| Linear prediction | `prediction = intercept + sum(coefficient_j * feature_j)`; coefficients act on the transformed features, not necessarily raw columns |
| Ridge | Minimize squared residuals plus an L2 coefficient penalty; stronger alpha penalizes large coefficients more |
| Logistic regression | A linear score passes through a sigmoid for binary probability modeling; C is an inverse regularization-strength control |
| Precision | TP / (TP + FP): reliability of predicted positives |
| Recall | TP / (TP + FN): coverage of actual positives |
| F1 | 2TP / (2TP + FP + FN); does not use true negatives directly |
| Decision tree | Splits feature space into regions using a training objective; deep trees can fit noise |
| Random forest | Combines randomized trees; aggregation reduces variance but does not prevent all overfitting/leakage |
| Boosting | Adds learners sequentially to improve a loss; learning rate/complexity/iterations interact |
| Kernel SVM | Uses feature-space similarities and a margin/regularization tradeoff; RBF gamma changes similarity locality |
| K-Means | Minimizes squared distances from records to assigned centroids |
| GMM | Fits a weighted mixture of Gaussian densities; components produce soft membership responsibilities |
| PCA | Finds high-variance directions in a centered numerical representation; component signs are arbitrary |
| LDA | Uses labels to favor class separation relative to within-class variation; binary output dimension is at most one |
| Bias/variance | Too-restrictive models underfit; flexible models can overfit; validation estimates behavior beyond fitting rows |
| Regularization | Limits model complexity relative to fit; not a substitute for honest evaluation |

Optional manual exercise: implement prediction and the squared-error objective for a one-feature linear model using NumPy; then derive/implement a gradient step. Compare with a library solution on tiny synthetic data. Keep this separate from the production experiment pipeline until tested. This deepens understanding without requiring you to rewrite forests or SVM solvers.

### 27.2 Milestone 1 checklist

- [ ] Team repository exists; all members have their own identities and real contributions.
- [ ] Clean setup/import/CLI path works.
- [ ] H2 source schema/size/quality audited; no H1 mixing.
- [ ] Structural and training EDA completed and explained.
- [ ] Prediction points and per-task feature policies documented.
- [ ] Frozen split manifests and preliminary leakage audit exist.
- [ ] Reusable preprocessing is fit only within training data/folds.
- [ ] At least one substantive untuned regression, classification and clustering experiment ran.
- [ ] Preliminary validation/internal metrics and interpretations are genuine.
- [ ] Supervised final test remains sealed.
- [ ] Progress report states completed work, limitations, contributions and remaining plan.
- [ ] Source ZIP contains required documentation/reproduction material and submitted SHA.

### 27.3 Final scientific checklist

- [ ] Three or more regression families, each baseline and tuned.
- [ ] Three or more classifiers, each baseline and tuned, including a kernel-based experiment.
- [ ] At least one supervised ensemble/boosting model.
- [ ] Three or more clustering algorithms, each initial and optimized with unsupervised criteria.
- [ ] Search spaces, budgets, criteria and best configurations recorded.
- [ ] MAE, MSE and/or RMSE, R-squared, validation/CV and final-test regression results.
- [ ] Accuracy, precision, recall, F1, confusion matrix, ROC-AUC and/or PR-AUC, CV and final-test classification results.
- [ ] PCA applied with retained-variance justification; two supervised models compared with/without PCA.
- [ ] One classifier compared on original, PCA and LDA representations.
- [ ] Clustering input/encoding/scaling justified; internal scores and cluster-count search.
- [ ] Two-dimensional visualization, feasible dendrogram and meaningful segment profiles.
- [ ] Complete leakage audit across features, fitting, tuning, cohorts and final evaluation.
- [ ] Winners selected using development evidence, with honest fair-comparison limitations.
- [ ] Saved complete pipelines and input schemas reproduce predictions.

The minimum baseline/tuned comparison is 12 supervised rows (3 models * 2 tasks * 2 stages) plus 6 clustering rows (3 algorithms * 2 stages), before extra representation variants, dummy references or additional protocols. Those rows can require hundreds of actual fits once CV/search is included.

### 27.4 Final application and release checklist

- [ ] Usable CLI or graphical interface; notebook-only submission avoided.
- [ ] Dataset analysis accessible.
- [ ] Training or explicit artifact loading accessible.
- [ ] Model comparisons and tuning results accessible.
- [ ] ADR and cancellation predictions use saved fitted pipelines.
- [ ] Clustering analysis is accessible.
- [ ] Selected model and relevant evaluation information displayed.
- [ ] Input errors and absent artifacts handled clearly.
- [ ] Inference does not refit.
- [ ] Environment specification/record and exact commands verified.
- [ ] Source/report/results ZIP excludes fitted binaries; artifacts available locally for discussion.
- [ ] GitHub URL/source SHA in submission README; actual history and contributions preserved.
- [ ] All five members can explain the complete workflow and its limitations.

### 27.5 Retrieval practice

Answer without looking, then check the expected principle:

| Question | Expected principle |
| --- | --- |
| Why not scale the full dataset before CV? | Scoring-fold observations would influence fitted preprocessing. |
| Does dropping ReservationStatus guarantee no leakage? | No; input timing, derived proxies and evaluation choices can still leak. |
| Why might "booking creation" be hard to claim? | Many exported values are later snapshots, not full event histories. |
| Does a new model require a whole new source tree? | No; a recipe/model factory can reuse shared implementation. |
| What does a source SHA fail to preserve? | Uncommitted code, untracked inputs, environment and data identity unless separately captured. |
| Why not use hard labels for ROC-AUC? | Ranking information is discarded. |
| Is a very high silhouette sufficient? | No; geometry, cohort, coverage, stability, sizes and meaning matter. |
| Does selecting a GPU accelerate the standard CPU pipeline? | No; the implementation/device must actually support GPU execution. |
| What must be saved besides coefficients? | Fitted preprocessing, feature/schema contract and task decision policy. |
| Why is final-test feedback dangerous? | It can turn the test into another model-selection set. |
| Why use your friends' Git commits for your Colab runs? | It preserves real authorship and exact source identity while you supply compute. |
| Can a polished UI compensate for invalid splits? | No; usability and scientific validity are separate requirements. |

## 28. Primary-source reading list

The course brief controls assessment. Keep a copy in your new repository. Official library documentation controls API behavior; use documentation matching the version recorded in your environment.

### Sources used for the guide's technical checks

- [Python virtual environments](https://docs.python.org/3.12/library/venv.html): environment creation, activation and interpreter behavior.
- [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/): pyproject/build-system/package concepts.
- [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow): branches, pull requests and collaboration.
- [Hotel booking demand datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/): original H2 source definitions and collection timing. Nuno Antonio, Ana de Almeida, Luis Nunes; Data in Brief 22 (2019), 41-49.
- [sklearn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): preprocessing consistency, leakage and randomness.
- [Mixed-type preprocessing example](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html): ColumnTransformer/Pipeline composition.
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): splitters and evaluation arrangements.
- [Hyperparameter search](https://scikit-learn.org/stable/modules/grid_search.html): search spaces, scorers, CV and refit.
- [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html): regression/classification/internal evaluation definitions.
- [PCA/decomposition](https://scikit-learn.org/stable/modules/decomposition.html): representation choices and explained variance.
- [LDA/QDA](https://scikit-learn.org/stable/modules/lda_qda.html): supervised discriminant analysis and solver behavior.
- [Clustering](https://scikit-learn.org/stable/modules/clustering.html): algorithms, assumptions, geometry and internal criteria.
- [Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html): mixture assumptions and model-selection criteria.
- [SciPy hierarchical linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html): dendrogram foundations and method constraints.
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html): environment compatibility and trusted artifact loading.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): account-private runtimes, sharing, persistence and resource limits.
- [XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/index.html): explicit device configuration and runtime requirements.
- [Streamlit first app](https://docs.streamlit.io/get-started/tutorials/create-an-app): the optional graphical route.
- [actions/checkout](https://github.com/actions/checkout) and [actions/setup-python](https://github.com/actions/setup-python): the CI template's action usage.

### Further study as you reach each topic

- [Python tutorial](https://docs.python.org/3/tutorial/): functions, modules, classes, errors, files and virtual environments.
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html): array shapes, vectorized operations and broadcasting.
- [pandas getting started](https://pandas.pydata.org/docs/getting_started/index.html): indexing, missingness and tables.
- [sklearn linear models](https://scikit-learn.org/stable/modules/linear_model.html): Ridge and logistic-regression objectives.
- [sklearn SVMs](https://scikit-learn.org/stable/modules/svm.html): kernels and practical scaling considerations.
- [Optuna tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html): the optional adaptive-search route.
- [uv projects](https://docs.astral.sh/uv/guides/projects/): an optional resolved-environment workflow.
- [Git book](https://git-scm.com/book/en/v2): deeper version-control concepts and recovery.

You do not have to understand every optional tool before starting. You do need to explain each tool, assumption, and transformation you actually choose.

### Validation record for the reference examples

Selected synthetic CPU examples were executed during preparation using Python 3.14 with NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1 and scikit-learn 1.9.1. Checks covered the package doctor example, preprocessing/unknown categories, the toy Ridge workflow, a training-statistics unittest, classification metrics, an eight-candidate three-fold search, PCA within CV, binary LDA, K-Means/GMM/agglomerative fitting, internal-score edge cases, and pipeline save/reload equality.

The submission-archive example was exercised with mocked Git responses and a synthetic source archive, including dotfiles and the raw-data directory placeholder; it preserved release metadata and rejected a fitted-model filename. This was not a release of an actual new repository.

All 27 Python code blocks passed syntax parsing; JSON/TOML and the CI YAML parsed; 19 PowerShell blocks passed syntax checks; and all 28 chapter-navigation links resolved. These checks do not claim that every context-dependent snippet, optional package route, platform installer, GPU configuration, live Colab runtime, future application, or CI run has been executed. No H2 experiment or real hotel performance result was produced. Your own implementation still needs its own tests and scientific validation.
