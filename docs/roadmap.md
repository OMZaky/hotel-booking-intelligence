# Delivery phases

Working dates are proposals; bold dates are from the supplied course PDF.

| Phase | Proposed dates | Deliverable / exit condition |
| --- | --- | --- |
| 0. Foundation | Oct 25–26 | Package, VS Code/Colab setup, recipes, CI; remote verified and environment lock validated |
| 1. Data and protocol | Oct 26–29 | H2 validation/EDA; approve prediction points and feature policies; freeze splits |
| 2. Milestone 1 baselines | Oct 30–Nov 4 | Shared preprocessing; one real model per task; validation results, audit and progress report |
| Milestone 1 | **Nov 5** | Submit ZIP, GitHub link, submitted SHA and contributions |
| Midterms | **Nov 6–12** | Light-work buffer; no course deadline or mandatory project discussion |
| 3. Model families | Nov 13–21 | At least three algorithms per task; fair baseline comparisons; ensemble and kernel coverage |
| 4. Tuning and representations | Nov 22–30 | Tune every algorithm; PCA/LDA comparisons; clustering interpretation |
| 5. Application and final evaluation | Dec 1–6 | CLI and agreed UI; saved-artifact inference; freeze recipes, then evaluate test set |
| 6. Release and report | Dec 7–10 | Clean-checkout reproduction, report, figures and rehearsal; prepare ZIP |
| Milestone 2 | **Dec 11** | Final submission with exact commands and submitted SHA |
| Preparation | Dec 12–17 | Verify local artifacts and rehearse complete workflow across five members |
| Live discussion | **Dec 18–24** | Demo and individual assessment; TA assigns exact slot |

## Data and first baselines

Keep raw bytes immutable and hash H2. Verify original row count, 31 source variables, dates,
categories, missingness, identifier semantics and duplicates. Do not drop duplicate records
without explaining what they represent. Define task-specific prediction points and allowlists.
Freeze split manifests before comparisons; review label availability for forward-looking claims.
Limit exploratory choices influencing selection to training/validation data.

Implement one complete slice: loading, reusable preprocessing, one substantive model per task,
validation metrics and report output. Dummy predictors provide additional sanity checks.
Keep the final supervised test set sealed through Milestone 1.

## Models, tuning and dimensionality reduction

Proposed families, to confirm after data/runtime inspection:

- Regression: Ridge/linear regression, random forest and gradient boosting.
- Classification: logistic regression, random forest and kernel SVM.
- Clustering: K-Means, GMM and sampled agglomerative clustering.

All nine selected algorithms require baseline and tuned/optimized comparisons.
Document search spaces, fixed budgets, primary metric, seeds, feature policy and split/cohort IDs.
Use a justified subset for expensive kernel/hierarchical experiments and report the comparison scope.
Run full-data K-Means where practical. Never use outcome labels to optimize clustering.
Report Silhouette, Davies–Bouldin, cluster sizes/stability, profiles and a two-dimensional view.
Include a dendrogram on a justified sample.

Compare at least two supervised models with/without PCA.
For at least one classifier compare original/PCA/LDA features.
Fit transformations inside CV folds. Binary LDA has at most one discriminant component.
Estimate dense-matrix memory after categorical encoding.

## Application and submission

CLI first; a Streamlit layer is a proposal for discussion. The application must expose analysis,
training/tuning access, comparisons, clustering results, and ADR/cancellation predictions from
saved complete pipelines. Inference loads artifacts and does not retrain at startup.

Report all required metrics, validation and final-test results, training times, search spaces,
best configs and model selection rationale. Select using validation; do not choose winners by
repeatedly checking final test scores. Retain local artifacts for the live discussion, outside ZIPs.
The report must explain the complete leakage audit and actual contributions of every member.

## Next decisions

1. Defensible operational prediction points for ADR and cancellation given the snapshot data.
2. CLI alone or CLI plus Streamlit.
3. Five member identities and rotating responsibilities.
4. Free Colab/local CPU and RAM budget.
