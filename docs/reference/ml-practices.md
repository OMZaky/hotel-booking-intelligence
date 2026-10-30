# Experimental practices

These conventions implement the brief. Exact feature/split policies require data inspection.

## Data and evaluation integrity

Define the operational prediction point before approving an explicit feature allowlist.
Use source semantics, not just correlation, to reject targets, future values and outcome proxies.
Freeze supervised holdouts before selection. Fit imputation, encoding, scaling, feature selection,
PCA and LDA inside training pipelines and inside each cross-validation fold.
Keep test data out of threshold selection, feature engineering and exploratory choices.
See [scikit-learn's guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Random and time-aware splits answer different questions. Choose the main claim, align split dates
with it, and account for label availability. An arrival-date split alone does not establish
booking-creation deployment validity. Document duplicate/group rules; absent customer IDs limit
claims that all customer overlap has been removed.

## Representations and runtime

Agent and Company are categorical IDs. Handle unknown categories at inference.
Scale distance-based/linear models as appropriate; keep sparse inputs where practical.
Estimate dense matrix memory before PCA/LDA after encoding. Tree and linear/distance models may
need different preprocessing; document each comparison rather than assuming one universal pipeline.

Start with CPU scikit-learn for this medium-sized tabular dataset. Add deep learning only for a
clear scientific reason. Avoid nested parallelism that oversubscribes cores across CV and estimators.

## Comparison records

Record task, model, recipe version, representation, baseline/tuned stage, seed, data hash, split ID,
source SHA, actual environment, parameters, timing and metrics. Use a common approved protocol,
bounded search budgets, and a declared primary metric. Quantify fold/seed variability where practical.

Regression reports MAE, MSE and/or RMSE, R², validation/CV and final test results.
Classification reports accuracy, precision, recall, F1, confusion matrix, ROC-AUC and/or PR-AUC,
CV and final test performance. Tune classification thresholds using validation data and stated costs.
Dummy predictors provide sanity checks; they do not replace required substantive model families.

## Clustering

Keep IsCanceled/final outcomes out of feature selection and tuning. Initial segmentation also
omits ADR conservatively; revisiting that needs a clear justification.
External outcomes may describe clusters after fitting, never select clustering hyperparameters.
Report Silhouette, Davies–Bouldin, cluster sizes/stability and profiles with representation,
distance metric, sample size and seed. Handle cases where internal scores are undefined.
Use justified samples for hierarchical clustering and pairwise scores; full K-Means where practical.
Separate descriptive segmentation from claims about predicting clusters for new bookings.

## Reproduction and inference

A seed is insufficient: preserve code, exact data bytes, splits, environment and complete recipe.
Generate a tested resolved environment lock before reported experiments; current dependency bounds
are not a lock. Every run also records installed versions.

Save the complete preprocessing/model pipeline and expected input schema after successful training.
Verify predictions before/after reload. Inference loads artifacts and never retrains at startup.
Only load trusted project artifacts in compatible environments; see
[scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).
