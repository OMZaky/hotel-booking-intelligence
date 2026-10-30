# Documentation

Start with the document that matches what you are doing. The two handbooks are
learning material for building independently in a **new repository**. The project
notes describe the **current bootstrap repository**, whose training and inference
features are still to be implemented.

## Learn and build by hand

1. Read the [course brief](reference/CSE382_Project_Description_Milestones.pdf)
   for the actual assignment requirements.
2. Work through [Build from scratch](guides/BUILD_FROM_SCRATCH.md) for installation,
   packages, Git/GitHub, team contributions, models, evaluation, and submission.
3. Use [Data preparation and feature engineering](guides/DATA_PREPARATION_AND_FEATURE_ENGINEERING.md)
   alongside the preparation chapters for detailed cleaning and feature decisions.
4. Use the [Colab guide](guides/colab.md) when running this repository remotely.
   For a new repository, adapt its paths and use the handbook's Colab instructions.

The handbooks contain synthetic worked examples. They are not evidence of trained
H2 models or a completed dataset audit. Their `booking_ml` package is an example
name; this repository's existing package is `hotel_ml`.

## Find a specific topic

| Folder | Document | Purpose |
| --- | --- | --- |
| `guides/` | [Build from scratch](guides/BUILD_FROM_SCRATCH.md) | Complete step-by-step learning route from an empty repository |
| `guides/` | [Data preparation and features](guides/DATA_PREPARATION_AND_FEATURE_ENGINEERING.md) | Ingestion, validation, cleaning, missingness, encoding, feature engineering, and safe pipelines |
| `guides/` | [Colab](guides/colab.md) | Existing launcher, revision selection, Drive persistence, and runtime limits |
| `project/` | [Architecture](project/architecture.md) | Existing implementation, proposed boundaries, and experiment/run identity |
| `project/` | [Roadmap](project/roadmap.md) | Course deadlines, proposed phases, and unresolved choices |
| `project/` | [Future structure](project/future-structure.md) | Deferred folders and the conditions for adding them |
| `reference/` | [Course brief](reference/CSE382_Project_Description_Milestones.pdf) | Authoritative assignment requirements and grading |
| `reference/` | [Leakage audit](reference/leakage-audit.md) | Snapshot timing, risky fields, and decisions required before training |
| `reference/` | [ML practices](reference/ml-practices.md) | Evaluation, preprocessing, comparison, and reproducibility conventions |
| `reference/` | [Sources](reference/references.md) | Primary technical and dataset references |

## What is authoritative?

The supplied course PDF defines assessment requirements. The roadmap's intermediate
dates and candidate model lists are proposals, not completed work. The architecture
describes this bootstrap; the handbooks explain possible implementations in a fresh
repository. Version-sensitive package/API examples reflect their stated preparation
date and should be checked when installing a different version.

For current commands and installation, start with the [repository README](../README.md).
For experiment recipes, see [configs](../configs/README.md). For user-supplied H2.csv,
see [data instructions](../data/README.md).

## Copying the learning material

Copy `guides/`, `project/`, `reference/`, and this index together to preserve links.
The guides can also be copied individually, but repair their companion/PDF links.
Replace example repository/package names with your own and record new scientific
decisions in your new repository. A new checkout does not contain the data,
generated artifacts, or a completed model implementation.
