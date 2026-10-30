# Primary research sources

Reviewed for setup on 2026-10-25. Course requirements/deadlines come from the supplied PDF.

| Source | Application |
| --- | --- |
| [Original H2 article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) | Scope, variables, engineering and snapshot timing |
| [scikit-learn pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Consistent preprocessing, training-only fitting and leakage |
| [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) | Holdouts and evaluation protocols |
| [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) | Source/data/artifact/report separation |
| [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) | GitHub loading, ephemeral runtimes and persistence |
| [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/) | Run metadata design; optional later tracking tool |

The structure adapts established conventions to three tasks and versioned recipes.
JSON manifests are the lightweight starting point; MLflow/DVC and hosted orchestration
are future options, not installed requirements.
