# Experiment recipes

Each JSON file is a complete recipe. Version the scientific recipe and code separately:

```text
configs/regression/v1/ridge-baseline.json
configs/regression/v1/ridge-tuned.json       # added when the search exists
configs/regression/v2/ridge-pca.json         # later feature/preprocessing change
```

Keep shared implementation in `src/hotel_ml/`; avoid copying model code into v1/v2 folders.
Every execution receives a separate run ID even when its recipe is identical.
Record model family, original/PCA/LDA representation, baseline/tuned stage, seed, split ID,
prediction point, primary metric, and estimator parameters.

The three starter configs are **setup examples, not approved training protocols**.
Their pending split/feature decisions must be resolved before training. Empty parameters mean
the library defaults; real tuning needs a documented, bounded search space in a future recipe.
Do not count a dummy predictor as one of the three required substantive model families.

Validate with `python -m hotel_ml validate-config --config <file>`.
Known direct targets/final status predictors are rejected when a feature list is supplied.
This guard is only a basic check: indirect leakage still needs the documented review.
