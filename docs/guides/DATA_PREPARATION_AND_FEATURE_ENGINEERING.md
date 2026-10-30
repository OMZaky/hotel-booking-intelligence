# Data cleaning, preparation, and feature engineering

A hands-on handbook for the **City Hotel (H2)** project, prepared on **28 October 2026**.

This guide is for learning and implementing the work yourself in a new repository. It does not depend on the implementation in the repository where you found it. The optional companion [build-from-scratch guide](BUILD_FROM_SCRATCH.md) covers the wider project, Git workflow, model training, application, and submission.

**Scope:** how to inspect raw bookings, decide what is valid and available, preserve the source, prepare model inputs, construct features, and prove that transformations behave correctly. There is no universal cleaning recipe or universally approved H2 feature list. Decisions depend on the prediction point, task, population, source evidence, and estimator.

**No H2 file is downloaded or analyzed here.** Examples are synthetic. Any numbers shown in worked examples describe those examples, not observed H2 findings.

## How to work through this handbook

For each topic: understand the failure it prevents, choose a policy, implement a small function, test a deliberately awkward example, record the decision, and only then apply it to your real development data.

Keep three separate records:

1. **Source audit:** what is present in the supplied file.
2. **Decision log:** what you chose, why, alternatives, and limitations.
3. **Fitted transformation record:** which statistics/categories/thresholds were learned from which training rows.

A dataset that looks tidy can still contain severe leakage. A transformation that runs without an exception can still answer the wrong question.

## Contents

1. [The mental model](#1-the-mental-model)
2. [Tools and packages](#2-tools-and-packages)
3. [Define the question before choosing inputs](#3-define-the-question-before-choosing-inputs)
4. [Preserve and import the source](#4-preserve-and-import-the-source)
5. [Understand all 31 H2 variables](#5-understand-all-31-h2-variables)
6. [Build a data-quality audit](#6-build-a-data-quality-audit)
7. [Clean strings and categories](#7-clean-strings-and-categories)
8. [Parse numbers and semantic types](#8-parse-numbers-and-semantic-types)
9. [Prepare dates correctly](#9-prepare-dates-correctly)
10. [Handle missing data](#10-handle-missing-data)
11. [Duplicates, identity, and dependence](#11-duplicates-identity-and-dependence)
12. [Outliers and unusual records](#12-outliers-and-unusual-records)
13. [Targets and row eligibility](#13-targets-and-row-eligibility)
14. [Split data and prevent leakage](#14-split-data-and-prevent-leakage)
15. [Encode categorical features](#15-encode-categorical-features)
16. [Transform and scale numeric features](#16-transform-and-scale-numeric-features)
17. [Design a feature contract](#17-design-a-feature-contract)
18. [Engineer H2 booking features](#18-engineer-h2-booking-features)
19. [Aggregates, histories, and target encoding](#19-aggregates-histories-and-target-encoding)
20. [Select and evaluate features](#20-select-and-evaluate-features)
21. [Choose task-specific preparation routes](#21-choose-task-specific-preparation-routes)
22. [Build a complete synthetic pipeline](#22-build-a-complete-synthetic-pipeline)
23. [Test preparation and inference](#23-test-preparation-and-inference)
24. [Version decisions and outputs](#24-version-decisions-and-outputs)
25. [Troubleshooting and bad examples](#25-troubleshooting-and-bad-examples)
26. [A practical learning sequence](#26-a-practical-learning-sequence)
27. [Final checklists and decision sheets](#27-final-checklists-and-decision-sheets)
28. [References and validation notes](#28-references-and-validation-notes)

## 1. The mental model

### 1.1 Four related activities

| Activity | Question | Examples |
| --- | --- | --- |
| Validation | Does this observation satisfy a declared contract? | Expected columns, valid target labels, finite ADR |
| Cleaning | Is there a justified correction or normalization? | Trim documented padding, parse declared NULL markers |
| Preparation | How must inputs be represented for this estimator? | Imputation, encoding, scaling, dense/sparse conversion |
| Feature engineering | What useful representation can be derived from available information? | Total nights, cyclic month, historical cancellation ratio |

Validation does not necessarily modify data. Cleaning should not invent facts. Preparation often learns state. Feature engineering can be deterministic, distribution-fitted, target-dependent, or time-dependent; those categories have different leakage risks.

### 1.2 The correct high-level flow

~~~text
immutable source bytes + file identity
    -> structural parsing and explicit schema checks
    -> justified row-level normalization and eligibility rules
    -> stable row identities and frozen split/cohort plan
    -> training-data exploration and decisions
    -> approved raw-feature allowlist
    -> deterministic feature construction
    -> fold-fitted imputation / encoding / scaling / selection
    -> estimator
    -> saved complete input-to-prediction pipeline
~~~

Some deterministic checks can run before splitting. Anything that estimates a distribution, vocabulary, threshold, target relationship, or selected feature set must respect training/fold boundaries.

### 1.3 Deterministic does not automatically mean safe

Computing TotalNights from two genuinely available fields is row-wise and does not require fitting. Computing a date from a final reservation status can also be deterministic but still leak a future outcome.

Likewise, an unsupervised transformation can leak: fitting PCA or an imputer on supervised test rows exposes that distribution to the model-selection procedure even without labels.

### 1.4 Keep source values separate from decisions

Do not overwrite the original CSV with a globally imputed or clipped export and call it the new truth. Preserve:

- original bytes;
- parsed but unmodeled records;
- quality findings;
- row inclusion/exclusion reasons;
- fitted transformation objects;
- task-specific matrices and their feature names.

The output for regression need not match classification or clustering. One global "cleaned.csv" is rarely the entire preparation contract.

**Checkpoint:** Explain the difference between correcting a misspelled category, imputing an unknown value, and engineering a ratio. Which ones can change the meaning of an observation?

## 2. Tools and packages

### 2.1 Minimal environment

Use your new repository's environment from the companion guide, or create one before practising. These examples target pandas **2.3.x** and scikit-learn **1.9.x**; the TargetEncoder example specifically uses a CV-splitter API available in 1.9.

Example installation into the intended environment:

~~~powershell
python -m pip install "pandas>=2.3,<3" "scikit-learn>=1.9,<2" numpy scipy
python -m pip install matplotlib seaborn joblib
python -m pip check
~~~

Here `python` must be your environment's interpreter. On Windows you can replace it with `.\\.venv\\Scripts\\python.exe`; on Linux/macOS use `.venv/bin/python` if you do not activate the environment.

### 2.2 What each package contributes

| Package | Role here |
| --- | --- |
| pandas | CSV parsing, labeled tables, missingness, grouping, dates, joins |
| NumPy | Arrays, finite-value checks, vectorized arithmetic, synthetic examples |
| scikit-learn | Fitted transformers, encoders, imputers, pipelines, feature selection and validation |
| SciPy | Sparse matrix support and scientific computations used by sklearn |
| Matplotlib | Explicit plotting and figure export |
| seaborn | Convenient statistical plots; optional |
| joblib | Save/reload your own trusted fitted pipelines |
| Ruff | Optional code formatting/linting |
| ipykernel | Optional local notebook kernel |

You do not install `csv`, `pathlib`, `json`, `hashlib`, `collections`, `datetime`, or `unittest`: they are part of Python.

Optional alternatives:

- **Pandera:** richer DataFrame validation schemas.
- **Great Expectations:** a larger data-quality framework; more setup.
- **PyArrow:** efficient columnar storage/Parquet, with dependency/version considerations.
- **imbalanced-learn:** resampling within a training pipeline; not generic cleaning.
- **category-encoders:** additional categorical encoders; study their leakage behavior.
- **holidays or an explicit calendar table:** optional holiday features; validate the calendar and prediction-time availability.

Do not install everything at once. An explicit pandas validator and standard sklearn pipeline are sufficient to begin.

### 2.3 Basic objects to understand

A DataFrame has row labels, column labels and dtypes. Semantic type and dtype differ: Agent may contain digits but is an identifier, not a measurement. A nullable integer column can contain unknowns; an ordinary NumPy integer array cannot represent NaN.

A transformer has:

- `fit`: learn state from the current fitting rows;
- `transform`: apply saved state;
- `fit_transform`: learn and apply, sometimes with special cross-fitting behavior;
- `get_feature_names_out`: describe output columns where supported.

Do not infer behavior only from the method name: TargetEncoder's fit_transform is deliberately different from fit followed by transform.

**Checkpoint:** Print your interpreter and package versions, then explain why a digit-only company code should not be standardized as a quantity.

## 3. Define the question before choosing inputs

### 3.1 Three task contracts

| Task | Outcome or purpose | Required boundary |
| --- | --- | --- |
| Regression | Predict ADR | ADR and outcome-derived price proxies cannot enter X |
| Classification | Predict eventual cancellation | IsCanceled and final-status consequences cannot enter X |
| Clustering | Discover interpretable booking patterns | Inputs and optimization must remain outcome-free |

Write down the prediction/analysis point, population, future horizon if relevant, and information available at that point. "Predict cancellation" alone is incomplete.

### 3.2 H2 has a source-timing limitation

The original paper describes many booking/change-log values relative to the day before arrival. The file is not a complete event history at booking creation.

Therefore:

- ReservedRoomType, arrival fields and party composition can look like creation-time fields while actually reflecting later amendments.
- AssignedRoomType, BookingChanges, waiting-list duration and mutable requests need operational timing review.
- DepositType was engineered from payments before arrival/cancellation; do not equate the exported value with an initial deposit contract.
- ADR is transaction-derived; an eventual daily-rate outcome is different from an initial price quote.
- Some bookings were canceled before a day-before-arrival decision point. They are not still at risk of a future cancellation then.

No amount of imputation, scaling or train/test splitting makes a future value retrospectively available.

### 3.3 Choose a defensible claim

Options include a carefully limited booking-creation proxy benchmark, a properly reconstructed later-snapshot study, or a clearly described retrospective prediction exercise. The source may not support a strong deployment claim.

Ask the instructor/TA when the intended operational claim cannot be established. State assumptions and limitations instead of declaring a feature list "leakage-free" merely because its validation score is good.

### 3.4 Build an availability table

For every feature and every task record:

~~~text
name | source | meaning | known by when? | include/exclude/quarantine
     | missing policy | transformation | rationale | remaining uncertainty
~~~

Use an allowlist for each task. A blacklist can fail when an accidental new label-derived column is added.

Derived features inherit the latest availability of their inputs. A room-change indicator does not become valid simply because you dropped AssignedRoomType after calculating it.

**Checkpoint:** For any engineered feature you propose, trace its source fields and explain the latest moment at which it could be computed.

## 4. Preserve and import the source

### 4.1 Preserve the original bytes

Keep your supplied H2 at `data/raw/H2.csv`. Ignore raw data in Git. Record a checksum:

~~~python
import hashlib
from pathlib import Path

source_path = Path("data/raw/H2.csv")
with source_path.open("rb") as stream:
    source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
print(source_hash)
~~~
This identifies bytes, not correctness. Reordered rows, encoding changes or a resaved CSV produce different bytes. Preserve the original and log every later interpretation.

### 4.2 Understand what CSV parsing can change

CSV parsing interprets delimiters, quotes, encodings, missing markers and types. A comma inside a quoted field is not a column separator. Splitting each line with `str.split(",")` is not a correct general CSV parser.

Do not use `on_bad_lines="skip"` to make malformed rows disappear. A parser can also pad short rows with missing fields; audit incomplete records and counts rather than assuming error mode catches every problem.

Choose the source encoding explicitly. `utf-8-sig` handles a UTF-8 BOM. If decoding fails, investigate the documented encoding; do not ignore decoding errors or choose a fallback only because it suppresses an exception.

### 4.3 A literal-first ingestion route

This route preserves textual tokens until you assign their meaning. Put this helper in a module you write, such as `preparation.py`:

~~~python
import csv
from collections import Counter
from pathlib import Path

import pandas as pd


def read_literal_csv(path, encoding="utf-8-sig"):
    path = Path(path)
    with path.open("r", encoding=encoding, newline="") as stream:
        try:
            header = next(csv.reader(stream))
        except StopIteration as error:
            raise ValueError("The CSV is empty.") from error

    if not header or any(not name.strip() for name in header):
        raise ValueError("Blank raw header.")
    duplicates = [name for name, count in Counter(header).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate raw headers: {duplicates}")

    return pd.read_csv(
        path,
        encoding=encoding,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        skip_blank_lines=False,
        on_bad_lines="error",
    )
~~~
Why this approach:

- Duplicate raw headers are checked before pandas can disambiguate names.
- All fields initially remain strings.
- Empty strings, "NULL", and "NA" are not silently interpreted as identical missing values.
- You choose missingness separately for each column.
- Blank/incomplete records remain visible to later auditing.

With `na_filter=False`, automatic NA token detection is disabled; `keep_default_na` is effectively not the controlling mechanism in that mode. It is included to make the literal-preservation intention clear.

**Alternative:** typed ingestion with `dtype` and column-specific `na_values`. It is convenient once the source contract is established. For example, preserve Agent/Company as strings while parsing numeric fields. Do not guess types/missing tokens before inspecting the source.

### 4.4 Canonicalize headers with an explicit mapping

~~~python
def canonicalize_headers(frame):
    original = list(frame.columns)
    canonical = [str(name).strip() for name in original]
    if any(not name for name in canonical):
        raise ValueError("Blank column name.")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Header normalization would create duplicate names.")
    result = frame.copy(deep=True)
    result.columns = canonical
    return result, dict(zip(original, canonical))
~~~
Trimming declared export padding is different from inventing a fuzzy mapping. If converting to snake_case, maintain a reviewed one-to-one mapping for all source labels. Do not automatically "fix" an unfamiliar column into the closest familiar spelling.

**Checkpoint:** Load a tiny CSV containing quoted commas, an empty field, "NULL", an ID with leading zeros and a duplicate header. Explain which stage should reject or interpret each case.

Source: [pandas 2.3.3 read_csv](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.read_csv.html).

## 5. Understand all 31 H2 variables

Expected original H2 size: **79,330 bookings, 31 source variables**. Verify the actual supplied file; size is not a substitute for schema validation.

| Source column | Semantic role | Cleaning/feature preparation questions |
| --- | --- | --- |
| IsCanceled | Binary outcome | Enforce 0/1 and target completeness; never impute labels |
| LeadTime | Duration in days | Nonnegative integer? skew? snapshot definition? log transform hypothesis |
| ArrivalDateYear | Calendar component | Valid integer year; source coverage versus future serving range |
| ArrivalDateMonth | English month name | Exact mapping, whitespace, missing/unknown spelling |
| ArrivalDateWeekNumber | Calendar week | Validate range; do not assume a universally 52-week year |
| ArrivalDateDayOfMonth | Calendar day | Must form a real date with month/year, not merely lie in 1-31 |
| StaysInWeekendNights | Count | Nonnegative integer; source meaning of planned/stayed nights |
| StaysInWeekNights | Count | Nonnegative integer; useful with weekend nights for totals |
| Adults | Count | Nonnegative? zero-adult records need investigation |
| Children | Count with possible missingness | Unknown is not automatically no children |
| Babies | Count | Validate integer/nonnegative; can affect party features |
| Meal | Nominal category | Source codes and Undefined/SC semantics; preserve meaning |
| Country | Nominal origin code | Missingness, validity of codes, cardinality and subgroup concerns |
| MarketSegment | Nominal category | Undefined categories, stable spelling, timing |
| DistributionChannel | Nominal category | Unknown values and consistency with source definitions |
| IsRepeatedGuest | Binary/history-derived | Validate 0/1; audit as-of construction |
| PreviousCancellations | Historical count | Nonnegative; history availability and denominator for ratios |
| PreviousBookingsNotCanceled | Historical count | Same timing issue; zero may mean no known history |
| ReservedRoomType | Nominal room code | Preserve codes; possible amendments |
| AssignedRoomType | Operational room code | Quarantine until prediction-time availability is defended |
| BookingChanges | Accumulated count | Future-event risk for early prediction |
| DepositType | Payment-derived category | Do not reinterpret as an always-known initial policy |
| Agent | Identifier/category | NULL meaning, leading zeros, cardinality; never mean-impute as an ID |
| Company | Identifier/category | Same; high missingness does not automatically make it useless |
| DaysInWaitingList | Duration/outcome of waiting | Timing review; nonnegative count if selected |
| CustomerType | Nominal booking type | Stable source categories and unknown handling |
| ADR | Regression target | Numeric, finite, complete; negative/zero/extreme values require domain review |
| RequiredCarParkingSpaces | Request count | Validate count; mutable/request timing |
| TotalOfSpecialRequests | Request count | Same; may proxy later engagement if unavailable initially |
| ReservationStatus | Final outcome/status | Audit separately; exclude from predictive/segmentation X |
| ReservationStatusDate | Final status time | Parse for auditing/label availability; exclude from predictive X |

Check names by set difference:

~~~python
def check_source_columns(frame, expected_names):
    expected = set(expected_names)
    actual = set(frame.columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if frame.columns.duplicated().any() or missing or unexpected:
        raise ValueError(f"Invalid source schema; missing={missing}, unexpected={unexpected}")
~~~
Construct `expected_names` from the exact table. For this source, it should have 31 distinct entries. A CSV with 31 wrong columns must not pass. Column order can be recorded without requiring the file to arrive in one arbitrary order; downstream name-based selection should be explicit.

If using a combined Kaggle-style mirror, isolate City Hotel first and document the complete source-name mapping. Do not merge resort/city records. An exported index column is a provenance issue to investigate, not automatically a feature.

**Checkpoint:** Distinguish a numeric count, a binary category, a nominal identifier and a date even when all four were originally stored as strings.

## 6. Build a data-quality audit

### 6.1 Measure before changing

Produce an audit with:

- source hash, parser/encoding options, row/column counts and source names;
- dtypes and semantic types;
- empty-string/NULL/other-token counts before normalization;
- missingness after declared normalization;
- unique-category counts and frequent/rare values;
- numeric ranges, quantiles, fractional counts, infinities and parse failures;
- date validity and coverage;
- exact duplicates, repeated feature patterns and any outcome disagreements;
- target completeness and valid labels;
- row eligibility/quarantine counts and reasons.

Do not run a huge correction chain first and then infer what the original problems must have been.

A small generic profile:

~~~python
def basic_profile(frame):
    return pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_count": frame.isna().sum(),
            "missing_fraction": frame.isna().mean(),
            "unique_nonmissing": frame.nunique(dropna=True),
        }
    )
~~~
On a literal-first frame, empty strings still count as values. That is intentional: record token frequencies first, then rerun the profile after explicit missingness normalization. `isna` cannot tell you the source meaning of a token you have not mapped.

### 6.2 Error severity

| Finding | Possible response |
| --- | --- |
| Absent target or wrong source schema | Stop and fix the source/mapping |
| Nonparseable count | Reject/quarantine with a recorded reason; do not hide it as ordinary missingness |
| Implausible but parseable duration | Investigate; may be an unusual real booking |
| Invalid calendar date | Reject/quarantine or use an explicit missing-date policy |
| Unknown category | Validate source meaning; establish inference fallback |
| Negative ADR | Investigate transaction/refund semantics; do not automatically set to zero |
| High missing Company | Compare meaningful absence handling, omission, or a category route |
| Identical records | Review booking identity and evaluation dependence |

### 6.3 What can inspect the whole file?

Structural integrity and provenance checks can establish what source was supplied. Use training data for exploratory choices that influence features, thresholds, categories, imputation, clipping or selection. Keep the final supervised test out of adaptive decisions.

A full-source category or missingness count must not silently become a fitted encoder vocabulary or a threshold chosen from test distribution. Separate audit findings from model-selection decisions.

**Exercise:** Write a quality issue table with columns `rule_id, severity, count, action, rationale`. Keep counts as observations, not as evidence that a particular cleaning action is correct.


### 6.4 Useful training-only diagnostic views

After semantic parsing and the split manifest exist, examine numerical distributions, missingness by column/time/category, category frequencies, and suspicious combinations. Keep the report clear about what denominator each rate uses.

~~~python
def numeric_profile(frame, columns):
    values = frame.loc[:, list(columns)]
    if len(values) == 0:
        raise ValueError("Cannot profile an empty cohort.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in values.dtypes):
        raise TypeError("Profile parsed numerical columns.")
    summary = values.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    summary["missing_fraction"] = values.isna().mean()
    summary["zero_fraction_all_rows"] = values.eq(0).fillna(False).mean()
    summary["negative_fraction_all_rows"] = values.lt(0).fillna(False).mean()
    return summary
~~~
For missingness by category, create a temporary missing flag and group it on the training cohort. A pattern can suggest a collection mechanism, but cannot prove MAR/MCAR/MNAR or justify a feature's timing.

Example plot, where `canonical_train` is your parsed, training-only frame:

~~~python
from pathlib import Path

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
lead = pd.to_numeric(canonical_train["LeadTime"], errors="raise")
axes[0].hist(lead.dropna(), bins=25)
axes[0].set(xlabel="Lead time (days)", ylabel="Bookings", title="Training lead time")
rates = (canonical_train.isna().mean() * 100).sort_values().tail(8)
axes[1].barh(rates.index.astype(str), rates.to_numpy())
axes[1].set(xlabel="Missing (% of training rows)", xlim=(0, 100))
fig.tight_layout()
destination = Path("reports/figures")
destination.mkdir(parents=True, exist_ok=True)
fig.savefig(destination / "training-quality.png", dpi=150)
plt.close(fig)
~~~
Use count/rate tables alongside plots. A high category mean can be based on very few records; report support. Correlations between encoded nominal numbers are not a substitute for meaningful categorical analysis. Do not quietly use test-set patterns to choose a cleaning policy.

## 7. Clean strings and categories

### 7.1 Preserve meaning while normalizing format

Possible problems include padding, inconsistent capitalization, invisible characters, alternate spellings and different missing tokens. A normalization should be based on the source contract, not on what makes a plot look convenient.

Keep codes such as reserved room types and company/agency IDs as nominal values. Leading zeros can be meaningful. Do not turn "0012" into integer 12 without a verified equivalence rule.

For free text, Unicode normalization can reconcile visually similar forms, but it can also merge distinct identifiers. Do not apply case-folding or Unicode normalization to every field indiscriminately.

### 7.2 A column-specific normalizer

~~~python
def normalize_text_columns(frame, columns, tokens_by_column):
    result = frame.copy(deep=True)
    for column in columns:
        values = result[column].astype("string").str.strip()
        tokens = set(tokens_by_column.get(column, ()))
        result[column] = values.mask(values.isin(tokens), pd.NA)
    return result
~~~
Example:

~~~python
example = pd.DataFrame(
    {
        "Agent": [" 0012 ", "NULL", ""],
        "Meal": ["BB", "Undefined", " SC "],
    }
)
normalized = normalize_text_columns(
    example,
    columns=["Agent", "Meal"],
    tokens_by_column={"Agent": ["NULL", ""], "Meal": [""]},
)
assert normalized.loc[0, "Agent"] == "0012"
assert pd.isna(normalized.loc[1, "Agent"])
assert normalized.loc[1, "Meal"] == "Undefined"
~~~
The function copies the input, strips declared padding, and maps only explicitly selected tokens. It does not decide that Meal="Undefined" means an absent record or that missing Agent proves there was no agency.

Audit how many cells changed for each rule. Preserve a mapping table when merging spellings. If two categories collapse, explain why they represent the same concept.

### 7.3 Unknown, missing, rare, and invalid are different

- **Missing:** no known value was supplied.
- **Unknown at inference:** a supplied value was not in the fitted vocabulary.
- **Rare:** a valid category had few training observations.
- **Invalid:** the value violates a declared source/input contract.

One reserved token should not accidentally collapse all four cases unless you deliberately choose and document that simplification.

An unseen agency might be legitimate. A numeric string in an English month field is a schema problem. A rare room code can be valid and important. Do not delete all low-frequency categories as "dirty."

### 7.4 External mappings

Mapping countries to regions or meal codes to business groups requires a versioned, reviewed lookup. Check unmapped values, many-to-one collapses, missing handling and timing. A dictionary downloaded today can contain categories/definitions that differ from the 2015-2017 source period.

Do not infer personal attributes from country or encode it into arbitrary numerical ranks. If used, inspect subgroup performance and uncertainty; deleting Country alone does not prove the remaining model is free of geographic proxies.

**Exercise:** Create a normalization table containing rule, affected column, old value, canonical value, rationale, changed count, and whether the rule is domain-fixed or learned.

## 8. Parse numbers and semantic types

### 8.1 Parsing is not imputation

`pd.to_numeric(errors="coerce")` converts failed parses to missing. That can be useful for detecting failures, but it is dangerous if the original invalid values are then treated as ordinary unknowns without any audit.

Track original missingness separately from newly failed parses. Do not silently turn a typo such as "two" into an imputed count.

### 8.2 An explicit count/measurement parser

This example accepts the declared decimal format and only blank/NULL missing tokens:

~~~python
import numpy as np


def parse_numeric_series(series, *, integer=False, minimum=None, maximum=None):
    text = series.astype("string").str.strip()
    text = text.mask(text.isin(["", "NULL"]), pd.NA)
    values = pd.to_numeric(text, errors="coerce")

    failed = text.notna() & values.isna()
    array = values.to_numpy(dtype="float64", na_value=np.nan)
    nonfinite = values.notna() & ~pd.Series(np.isfinite(array), index=series.index)
    if failed.any() or nonfinite.any():
        raise ValueError(f"Invalid numeric input in {series.name!r}.")

    if minimum is not None and (values < minimum).fillna(False).any():
        raise ValueError(f"Below minimum in {series.name!r}.")
    if maximum is not None and (values > maximum).fillna(False).any():
        raise ValueError(f"Above maximum in {series.name!r}.")

    if integer:
        if (values.abs() > 2**53 - 1).fillna(False).any():
            raise ValueError("Use a different exact-integer contract for huge values.")
        if (values % 1 != 0).fillna(False).any():
            raise ValueError(f"Fractional count in {series.name!r}.")
        return values.astype("Int64")
    return values.astype("Float64")
~~~
Example:

~~~python
counts = parse_numeric_series(
    pd.Series(["2", "0", "NULL"], name="Children"),
    integer=True,
    minimum=0,
)
assert counts.iloc[0] == 2
assert pd.isna(counts.iloc[2])
~~~
This helper is appropriate for ordinary counts/measurements. It is not an identifier parser. It deliberately rejects very large integer quantities rather than risking float precision loss.

The helper permits missingness; a separate target/required-field contract must reject missing values where necessary.

### 8.3 Range, integrality, and semantics

- Children=1.5 parses numerically but violates an integer-count contract.
- IsCanceled=2 is an integer but violates the binary outcome contract.
- Agent="123" is an identifier even though it is numeric-looking.
- ADR=-5 is a parseable measurement; its domain validity requires a decision, not automatic zeroing.
- A million parking spaces is nonnegative but implausible; plausibility and hard impossibility are different checks.

Use nullable `Int64` for counts with missing values during pandas preparation. Use `Float64` or regular float arrays for numerical modeling, with an explicit pandas-to-sklearn boundary. Do not call ordinary `astype(int)` before deciding how to handle unknown counts.

### 8.4 Locale and unit problems

A value "1,200" can mean twelve hundred or one point two depending on formatting conventions. A decimal comma in a quoted CSV field is not solved by removing every comma everywhere.

If the source requires thousands/decimal separators, define and validate that format per column, then parse it. Record units: days versus hours, nights versus stays, rates versus totals. No scaler repairs a duration that was misread in the wrong unit.

**Checkpoint:** Explain the difference between parsing failure, legitimate missingness, impossible value and unusual valid value.

Source: [pandas 2.3.3 to_numeric](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.to_numeric.html).

## 9. Prepare dates correctly

### 9.1 A date must exist on the calendar

Day=31 and month=February each satisfy naive separate ranges, but their combination is invalid. Leap years matter. Date-only H2 fields do not contain precise booking-event timestamps or time zones.

A strict helper for the source's three arrival components:

~~~python
MONTH_NUMBER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def make_arrival_dates(frame):
    years = parse_numeric_series(frame["ArrivalDateYear"], integer=True, minimum=1)
    days = parse_numeric_series(
        frame["ArrivalDateDayOfMonth"],
        integer=True,
        minimum=1,
        maximum=31,
    )
    months = frame["ArrivalDateMonth"].astype("string").str.strip().map(MONTH_NUMBER)
    parts = pd.DataFrame(
        {
            "year": years.to_numpy(dtype="float64", na_value=np.nan),
            "month": months.to_numpy(dtype="float64", na_value=np.nan),
            "day": days.to_numpy(dtype="float64", na_value=np.nan),
        },
        index=frame.index,
    )
    dates = pd.to_datetime(parts, errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid or incomplete arrival-date components.")
    return dates
~~~
This route requires a complete valid date. An alternative policy can propagate NaT plus a missing-date indicator, but it must specify how later calendar features behave. Do not silently substitute today's date or the final status date.

If normalizing English month capitalization, make that a declared text rule. Unrecognized language/spelling should produce an explicit finding rather than an arbitrary month number.

### 9.2 Validate related fields

ArrivalDateWeekNumber may follow a specific source convention. Verify that convention before comparing it with pandas ISO week output. ISO week-year can differ from calendar year around January/December; a mismatch is not automatically corrupted data.

Check date coverage against the supplied H2 population, but distinguish a **source-period audit** from a **serving contract**. Hard-coding that all future user inputs must be in 2015-2017 would make a purported future prediction app unusable without explaining its limited scope.

### 9.3 Dates that are available versus dates that reveal outcomes

ArrivalDate can be a predictor/split field under an appropriate policy. ReservationStatusDate is a final outcome timestamp. Parse it separately for auditing and label availability, not as a convenient predictor.

Approximate booking-entry date:

~~~python
# Only a proxy if the exported arrival/lead-time values reflect later amendments.
booking_date_proxy = arrival_dates - pd.to_timedelta(lead_time_days, unit="D")
~~~
Here `arrival_dates` and `lead_time_days` must be validated aligned Series. The expression does not reconstruct unavailable booking history.

### 9.4 Calendar features

Month number is ordered but December and January are adjacent in seasonality. Possible representations:

- one-hot month;
- sine/cosine month pair;
- day of week;
- justified trend/year field;
- known holiday/business-calendar indicators.

Do not assume every week cycle has exactly 52 slots. For day-of-year features, account for leap years or use a deliberately chosen calendar convention. Calendar information known in advance can be row-wise; seasonality estimates learned from outcome patterns must be fitted on development data.

**Checkpoint:** Test 2016-02-29, 2015-02-29, January year boundaries, missing month, and an unknown month name.

Source: [pandas 2.3.3 to_datetime](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.to_datetime.html).

## 10. Handle missing data

### 10.1 Identify meaning before choosing a method

Missingness can mean unknown, not collected, not applicable, or a source error. For Agent/Company, absence may reflect no associated entity rather than an unknown real ID; verify that interpretation instead of assuming it.

Three statistical descriptions:

- **MCAR:** missingness does not depend on observed or unobserved values.
- **MAR:** after conditioning on observed variables, missingness does not depend on the missing value.
- **MNAR:** missingness depends on unobserved information/the missing value even after conditioning.

You generally cannot prove these mechanisms from a few plots. Use them to state assumptions and sensitivity questions, not to attach a confident label to every column.

### 10.2 Method comparison

| Method | Possible use | Main limitation |
| --- | --- | --- |
| Drop rows | Missing indispensable target/input with justified population restriction | Changes population, class proportions and sample size |
| Drop a column | Unavailable, uninformative, or unusable under the prediction question | High missingness alone is not proof of uselessness |
| Mean | Numeric roughly symmetric distribution | Sensitive to extremes; shrinks variance |
| Median | Robust numeric starting route | Ignores conditional relationships; still creates an artificial mass |
| Most frequent | Some categorical settings | Can erase meaningful absence and amplify the dominant category |
| Constant category | Explicit missing/not-applicable representation | Requires documented semantics and collision-free sentinel |
| Constant numeric | Only when a domain-fixed value is justified | Zero can be wrong and may look like a real quantity |
| KNNImputer | Local numerical relationships are meaningful | Distance/scaling/cost and categorical compatibility |
| IterativeImputer | Multivariable numerical estimation is worth complexity | Assumptions, convergence and additional variance; experimental sklearn API |
| Native missing handling | Chosen estimator actually supports the missing form | Support depends on estimator/version; preprocessing may still reject it |

For IterativeImputer, the optional import enabling its experimental API is required before importing the class. Check its documentation and test stability. Do not turn categories into arbitrary ordered numbers merely to force them through a numerical imputer.

Do not use ADR or IsCanceled to fill predictor values in a way that requires knowing them for a new booking. Supervised relationships used in feature construction need a valid train/inference contract, not a magically available outcome.

### 10.3 Fit the imputer on fitting rows only

A median is learned state. During CV it must be re-estimated separately for each fitting fold, not once on all development rows before CV.

For a frozen final pipeline: fit the chosen imputer on the declared training/development data and call transform on new data. Do not recompute medians independently in the app.

### 10.4 Missing indicators

Missingness can carry useful information, but it can also encode a data-collection artifact that shifts later.

With SimpleImputer(add_indicator=True), indicators are normally based on features missing during fitting. A feature newly missing only at inference may not receive a new indicator column. Variable output width at inference would be invalid.

For an always-declared indicator set, use a documented fixed-feature strategy. Example:

~~~python
from sklearn.impute import MissingIndicator

indicator = MissingIndicator(features="all", error_on_new=False)
indicator.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
new_flags = indicator.transform(np.array([[np.nan, 5.0]]))
assert new_flags.tolist() == [[True, False]]
~~~
You can also add row-wise missing flags for specific raw inputs in your own feature builder. Keep their names/meaning stable; do not add columns opportunistically only when a batch happens to contain missing values.

### 10.5 All-missing and constant columns

A training fold can contain an entirely missing column even if the whole dataset does not. With the appropriate SimpleImputer settings, `keep_empty_features=True` preserves its output position. A numeric software fallback such as zero does not prove zero has a valid business meaning.

Choose: preserve with an explicit fallback/flag, remove using training-only rules, or reject the feature policy. Test held-out values in that column after a fold with no observations.

A column constant in training can vary in new data. Decide whether it is retained, removed by a fitted selector, or treated as an unsupported input change.

### 10.6 Quantify consequences

Report missing rates, affected records, imputation policy, indicator policy and sensitivity comparisons. Compare median versus alternatives through development validation rather than declaring a winner from final test results.

**Exercise:** A missing Children value plus Adults=2 does not prove a two-person party. Show how different imputation choices change TotalGuests, and state which fact is observed versus invented.

Sources: [pandas missing data](https://pandas.pydata.org/pandas-docs/version/2.3.3/user_guide/missing_data.html), [sklearn imputation](https://scikit-learn.org/stable/modules/impute.html).


## 11. Duplicates, identity, and dependence

### 11.1 Different meanings of "duplicate"

| Pattern | What it establishes | What it does not establish |
| --- | --- | --- |
| Every exported field identical | Two rows have identical snapshots | They are the same real booking |
| Approved predictors identical | Model sees indistinguishable inputs | Outcomes must be identical |
| Same real customer identifier | Potential repeated-customer dependence | Identifier is safe to expose as a feature |
| Same date/party/category combination | Shared observable pattern | Same person or booking |

Identifiers were removed from the published data. You cannot reconstruct customer identity reliably from a repeated-guest flag or a few booking attributes.

`frame.duplicated(keep=False)` marks all members of repeated patterns. `frame.duplicated()` marks later occurrences beyond the first. Those counts answer different questions.

### 11.2 Do not use deduplication to improve scores

Investigate why duplication exists before choosing:

- preserve legitimate booking multiplicity;
- remove only proven accidental duplicated exports;
- group identical predictor patterns across splits;
- report sensitivity to a documented deduplication policy.

Dropping every duplicate can change booking frequencies and class/target distributions. Deduplicating using outcomes can also make the study population depend on future information.

If identical predictors have different targets, the inputs may simply be insufficient to distinguish outcomes. Do not delete the inconvenient label to create a clean relationship.

For clustering, repeated observations influence density/centroids. That can be appropriate for booking-volume segmentation. Deduplication changes the geometry/population; not all algorithms support compensating sample weights.

### 11.3 Preserve stable row identity

Use a metadata key such as source SHA plus parsed-record ordinal. Preserve it through quality flags and eligibility masks. It belongs in manifests/audit tables, not in model X.

A record ordinal is not necessarily a physical CSV line number because quoted fields can contain newlines. An exported row number may also encode time/order and must not become an accidental predictor.

When rows are excluded, store their original keys, rule IDs, counts and rationale. Do not reset indices and then apply a split manifest made for a different row order.

### 11.4 Grouping without overclaiming

A hash of approved predictor values can group exact feature patterns. It is not a customer ID. Hash implementations can be version-dependent and collisions are possible; use them with a documented equality/grouping policy.

For true group claims, use a defensible source grouping key. If grouping is required, preserve it in both the outer split and CV. Fixing only one boundary still allows dependent rows across other fitting/scoring partitions.

**Checkpoint:** Explain whether your unit is a booking, customer, group booking, or feature pattern. Which of those units can H2 actually identify?

## 12. Outliers and unusual records

### 12.1 Unusual is not necessarily erroneous

A large party, long lead time or expensive booking can be real. Outliers can reveal data errors, rare legitimate behavior, unit mistakes, population mixtures, or target-definition issues.

Separate:

1. impossible under a verified domain rule;
2. suspicious and requiring source investigation;
3. unusual but plausible;
4. influential for a particular estimator.

Do not remove a record only because a model predicts it badly.

### 12.2 Detection approaches

| Approach | Appropriate question | Limitation |
| --- | --- | --- |
| Domain bounds | Does it violate a verified business/data rule? | Invented bounds remove valid cases |
| Quantiles/IQR | Is it far into this training distribution's tail? | Skew and legitimate extremes are not errors |
| Standard z-score | Is it far relative to mean/std? | Sensitive to skew/extremes; requires assumptions |
| Median/MAD | Is it far relative to a robust center/spread? | Zero MAD and discrete counts need handling |
| Isolation/local-density methods | Is it unusual in multivariable geometry? | Hyperparameters, scaling and cohort determine meaning |
| Residual analysis | Where does this fitted model fail? | A diagnostic, not automatic source correction |

The familiar 1.5*IQR rule is a screening convention, not a universal deletion rule. Never estimate thresholds from the supervised final test.

### 12.3 Response alternatives

- Keep valid extremes and use an appropriate loss/model.
- Correct a proven unit/entry error with provenance.
- Quarantine unsupported records with reason codes.
- Transform a skewed variable.
- Clip/cap using training-fitted bounds.
- Add a stable out-of-range indicator if justified.
- Define a narrower population explicitly and report coverage/limitations.

Clipping changes values and often the target relationship. Capping counts at quantiles can produce fractional transformed values; do not overwrite original count fields and imply they were observed that way.

### 12.4 A training-fitted quantile clipper

This example illustrates learned-state placement. It is an optional hypothesis, not a required H2 cleaning step. Put custom transformers in an importable module.

~~~python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TrainingQuantileClipper(TransformerMixin, BaseEstimator):
    def __init__(self, columns, lower=0.01, upper=0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a DataFrame.")
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Invalid quantile limits.")
        if not isinstance(self.columns, (list, tuple)) or not self.columns:
            raise ValueError("Declare a nonempty list/tuple of clipping columns.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Clipping columns must be distinct.")
        values = X.loc[:, list(self.columns)].astype("float64")
        if np.isinf(values.to_numpy()).any():
            raise ValueError("Infinity is not a valid clipping input.")
        lower_bounds = values.quantile(self.lower)
        upper_bounds = values.quantile(self.upper)
        if lower_bounds.isna().any() or upper_bounds.isna().any():
            raise ValueError("A clipping column has no usable training values.")
        self.lower_bounds_ = lower_bounds
        self.upper_bounds_ = upper_bounds
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a DataFrame.")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Input columns/order changed.")
        result = X.copy(deep=True)
        for column in self.columns:
            values = pd.to_numeric(result[column], errors="raise").astype("float64")
            if np.isinf(values.to_numpy()).any():
                raise ValueError("Infinity is not a valid clipping input.")
            result[column] = values.clip(
                lower=self.lower_bounds_[column],
                upper=self.upper_bounds_[column],
            )
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("Unexpected input feature names.")
        return self.feature_names_in_.copy()
~~~
Learn bounds on the current fitting rows; transform validation/test/inference using those fixed bounds. The class preserves row count, index and column names, and copies input data.

If all training values for a column are missing, this implementation raises rather than inventing a quantile. Choose a deliberate alternative policy if needed.

For a comparison, try no clipping versus a small predeclared bound set inside CV. Record how many training/validation values are affected without recalculating bounds from validation.

### 12.5 Do not hide target errors with preprocessing

ADR is y for regression. A predictor clipper should never receive the target column. Target winsorization or row removal changes the regression problem and requires its own explicit study/population contract.

Changing a valid extreme y to a convenient value can make evaluation look better while misrepresenting the original objective. Robust losses and separate error analyses may be preferable.

**Checkpoint:** Fit the clipper on [0,10,20], transform 100000, and prove its bounds did not change. What happens if all training values are missing?

## 13. Targets and row eligibility

### 13.1 Validate labels separately

For classification, parse IsCanceled as an integer and check membership in {0,1}. For regression, parse ADR as numeric, reject infinity, and investigate missing/unusual values.

Never median-impute ADR or fill missing IsCanceled with zero. An unavailable label is not a normal predictor-missingness problem.

Do not overwrite IsCanceled by guessing how ReservationStatus="No-Show" should be interpreted. Investigate the source definition and any inconsistencies instead.

### 13.2 Define the prediction population

Possible ADR studies include all bookings with valid defined ADR or a justified subset such as completed stays. These ask different questions.

If a subset is selected using a future outcome, acknowledge that it is a conditional retrospective population that may not be identifiable at prediction time. Do not claim an all-booking deployment model was evaluated if you removed future cancellations using their labels.

Similarly, a cancellation study at a later decision point must establish which bookings were still active then. Source snapshots may not provide sufficient history.

### 13.3 Separate three kinds of row rules

| Rule | Example | Evidence needed |
| --- | --- | --- |
| Structural invalidity | Cannot parse required date | Parser/schema contract and recorded reason |
| Scientific eligibility | A declared cohort/time window | Question and predeclared protocol |
| Learned outlier/filter rule | Training-estimated threshold | Fitting-only estimation and honest scoring coverage |

Row removal inside an ordinary sklearn transformer is generally unsafe: X, y, groups, weights and split indices can become misaligned. Determine eligible rows in a documented data/cohort layer, or use a carefully designed method that coordinates every aligned array.

Do not discard invalid scoring rows silently and report a metric only for easy cases. State failures and coverage; define beforehand how unsupported inputs affect evaluation.

### 13.4 Class imbalance is not dirty data

A higher cancellation/non-cancellation prevalence does not mean you should equalize the raw dataset globally.

Consider stratified splitting, class weights, threshold policies, and appropriate metrics. If resampling, do it inside fitting folds. Leave validation/test distributions representative of the declared question rather than balancing them to make scores convenient.

**Checkpoint:** For every removed row, distinguish "invalid source" from "not in the studied population" from "hard for the model."

## 14. Split data and prevent leakage

### 14.1 What can be decided before the split?

Source identity, declared schema, source-fixed token mappings and row-wise verified validity rules can be established before splitting. Source-fixed does not mean invented after seeing performance.

Training-derived medians, quantiles, categories, rare thresholds, target associations, feature rankings and representation parameters must be fitted within the current training partition/fold.

### 14.2 Choose the evaluation design

- **Random/stratified benchmark:** useful for an exchangeable-booking question; not proof of future deployment.
- **Grouped split:** use when a real dependency unit must stay together.
- **Chronological split:** use whole time/cohort boundaries for forward-looking claims.
- **Hybrid:** groups plus temporal rules and label-availability purging when the claim requires them.

Persist the plan before repeated experimentation. Store raw hash, included original row keys, split/cohort IDs, seed, strategy, loader/normalization identity, and relevant time/group rules.

### 14.3 Inner folds matter too

If a grouped outer split is used, an ordinary random inner CV can still leak dependent groups. A forward split also needs label-availability rules inside the search folds.

For temporal evaluation, check both input availability and when training outcomes were resolved. An earlier-created booking with a later-resolved label may be unavailable for an earlier fitting window.

TimeSeriesSplit does not automatically enforce whole-date boundaries, grouped duplicates, correct prediction time or label-resolution purging.

### 14.4 What leakage looks like in preparation

Unsafe examples include:

- imputing all rows, then splitting;
- fitting an encoder's rare-category counts on train plus test;
- clipping using full-source quantiles chosen for model performance;
- selecting features using correlations/importance computed on validation/test repeatedly;
- fitting PCA/LDA before cross-validation;
- target encoding training rows using their own labels;
- aggregating "historical" outcomes from events not yet resolved;
- scaling a clustering cohort that later influences supervised choices through held-out outcomes.

Even using labels only for feature selection leaks if selection crosses the wrong evaluation boundary.

### 14.5 Do not make processed data the hidden boundary

If you export a preprocessed matrix, also record:

- exactly which source rows produced it;
- which fitted transformer produced it;
- training split/fold identity;
- feature names/order and types;
- whether it is training out-of-fold or holdout/inference representation;
- source code/configuration identity.

A single globally processed CSV cannot generally be reused for honest CV with learned preparation. Caching per-fold transformations is a performance option only if cache keys preserve these boundaries.

**Checkpoint:** Draw your pipeline for one CV fold and mark every operation that learns from rows or labels.


## 15. Encode categorical features

### 15.1 Match representation to meaning

Nominal categories do not have an intrinsic order. Labeling countries 0,1,2 creates arbitrary distances/orderings if treated numerically. A categorical ID can be useful through repeated associations, but also overfit, shift, or encode undesirable proxies.

Choose an encoder together with the estimator and evaluation protocol.

| Encoding | Benefit | Main concern |
| --- | --- | --- |
| One-hot | Clear nominal indicators | High dimension; rare/unseen categories; sparse versus dense |
| Ordinal | Compact codes | Artificial order/distances unless estimator handles categories appropriately |
| Frequency/count | Compact training-distribution summary | Must fit on training; changing population; not a historical count automatically |
| Hashing | Fixed dimension, no learned vocabulary | Collisions and reduced interpretability |
| Target encoding | Compact supervised category signal | Self-label leakage and as-of/group/CV complications |
| Native categorical handling | Model-specific category treatment | Version, cardinality, missing/unseen and persistence constraints |

### 15.2 One-hot encoding

A fitted encoder learns category vocabulary from fitting rows. The same raw category must map to the same output position at prediction time.

~~~python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
encoder.fit(pd.DataFrame({"Meal": ["BB", "HB", "BB"]}))
encoded_new = encoder.transform(pd.DataFrame({"Meal": ["BB", "NEW"]}))
assert encoded_new.shape[0] == 2
assert encoded_new.toarray()[1].sum() == 0
~~~
This tiny dense conversion is for inspection only. Do not densify the full H2 matrix casually.

With `handle_unknown="ignore"`, unknown values use an all-zero indicator vector for that feature. With `drop="first"`, a reference category can also be represented by zeros, making that distinction less explicit. Dropping a category can help coefficient interpretation for some linear designs, but is not a universal requirement for regularized models.

### 15.3 Rare-category grouping

`min_frequency` or `max_categories` can group rare training categories. Learn the vocabulary/groups within fitting folds.

`handle_unknown="infrequent_if_exist"` maps unseen values to an infrequent bucket **if that bucket exists**. If no infrequent bucket was learned, unknown handling behaves like the ignore route. Do not promise a separate "other" output column always exists.

Example candidate:

~~~python
grouped_encoder = OneHotEncoder(
    handle_unknown="infrequent_if_exist",
    min_frequency=2,
    sparse_output=True,
)
grouped_encoder.fit(pd.DataFrame({"Agent": ["A", "A", "B", "C"]}))
~~~
Two is a teaching value, not a proposed real-data optimum. Tune/choose grouping from development evidence and source meaning. A globally rare category may be common in one training fold; that is a normal fitted-state difference.

### 15.4 Alternative routes

**OrdinalEncoder:** use explicit unknown/missing policies if the model accepts the resulting numerical matrix. A negative unknown code is a marker, not a country "below" another country. Ordinary KNN/K-Means/PCA will interpret the codes geometrically unless you change the method.

**Frequency encoding:** a category maps to its count/frequency in the fitting cohort. It can confuse equally frequent categories and changes under population shift. Provide a deterministic unseen fallback and optional unknown indicator.

**Hashing:** fixes width and handles unseen strings without a vocabulary. Collisions mix categories. Hashing does not make an outcome-derived input acceptable; it can hide a leak behind opaque columns.

**Native categorical estimators:** preserve a category-aware representation and follow that estimator's documented constraints. sklearn forests, histogram boosting, XGBoost and other libraries do not all accept raw strings/categorical dtypes in the same way. Verify the selected implementation.

**Target encoding:** covered separately in Chapter 19 because label-dependent preparation requires special handling.

### 15.5 pandas/sklearn boundary

Pandas StringDtype can carry pd.NA. For a chosen sklearn pipeline, normalize to the expected missing representation:

~~~python
def categorical_boundary(frame, columns):
    result = frame.copy(deep=True)
    for column in columns:
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result
~~~
Apply the same conversion at training and inference. Do not maintain two incompatible vocabularies or let one environment treat "NULL" as a category while another treats it as missing.

**Checkpoint:** Explain what happens for a missing value, a rare known category and an entirely unseen category. Show the actual output names/dimension.

Source: [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

## 16. Transform and scale numeric features

### 16.1 Scaling changes geometry and regularization

| Method | What it does | Good starting use / limitation |
| --- | --- | --- |
| StandardScaler | Subtract training mean, divide by training standard deviation | Linear/kernel/distance/PCA routes; sensitive to extremes |
| RobustScaler | Use training median and interquartile range | More robust location/scale; does not remove outliers |
| MinMaxScaler | Map training range to a chosen interval | New values can fall outside it; clipping is an explicit extra policy |
| MaxAbsScaler | Divide by training maximum absolute value | Can preserve sparse zeros; sensitive to a large maximum |
| Normalizer | Normalize each row's vector length | Changes meaning from total magnitude toward relative pattern |
| No scaling | Keep units | Often appropriate for tree splits; not a safe distance default |

Scaling numerical branches before concatenating one-hot indicators is different from scaling every encoded indicator too. Standardizing rare binary indicators can give them large effective weights. For clustering, document the relative weight of numeric and categorical feature blocks.

Do not center a large sparse matrix with the ordinary mean-centering route: it destroys sparsity or errors. Choose a sparse-compatible policy such as appropriate `with_mean=False`, or keep numeric scaling inside a dense numerical branch before combination.

### 16.2 Skew transformations

**log1p:** for a chosen nonnegative variable, `log1p(x)` handles zero without adding an arbitrary large constant. Validate the domain. A log transform changes differences: 10-to-20 no longer has the same transformed increment as 100-to-110.

**PowerTransformer:** Yeo-Johnson can handle positive and negative values; Box-Cox requires strictly positive input. The transformation learns parameters from fitting rows, and sklearn's default standardization is an additional part of the transformation.

**QuantileTransformer:** learns a training empirical distribution and maps ranks toward a uniform or normal distribution. It is nonlinear, handles tails differently and can distort original distances/relationships. Fit only on training; investigate ties and behavior outside the observed range.

A log or rank transform does not repair a value in the wrong unit or an impossible count.

### 16.3 Scaling is not a normality requirement

Many models do not require normally distributed X. A prettier histogram is not automatically a better predictive representation. Evaluate the transformed route on the declared development protocol and report effects on interpretation and inference.

ADR target transformation is a separate decision from transforming an input. If modeling transformed y, preserve the inverse transformation in a suitable target wrapper and evaluate in original outcome units. Check the target domain first.

### 16.4 Ordering matters

A recommended simple numerical branch is median imputation followed by scaling. Other routes may need a different order:

- a skew transform requires validated domain values;
- KNN imputation depends on distances, so scale choice/missing-value support must be designed together;
- clipping before a derived logarithm differs from clipping only one of several redundant derived features;
- an all-missing column needs a declared fallback before methods unable to estimate it.

If a feature builder retains both raw LeadTime and LogLeadTime, clipping only LeadTime later does not bound the information in LogLeadTime. This is not target leakage by itself, but it may contradict your stated outlier policy.

**Exercise:** Compare unscaled, standard-scaled and robust-scaled toy durations. Explain which values/statistics changed and which were learned.

Sources: [preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html), [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html).

### 16.5 Binning and discretization

Binning replaces a numerical value with an interval/category. It can express a business rule or reduce sensitivity, but discards distinctions and introduces boundary effects.

**Domain-fixed bins:** the edges are justified before seeing model scores, so assigning a row to a bin is deterministic.

~~~python
lead_for_bands = pd.Series([0.0, 6.0, 7.0, 30.0, 90.0, np.nan])
lead_bands = pd.cut(
    lead_for_bands,
    bins=[0, 7, 30, 90, np.inf],
    right=False,
    labels=["under_week", "week_to_month", "month_to_quarter", "long_lead"],
)
assert lead_bands.iloc[2] == "week_to_month"
assert lead_bands.iloc[4] == "long_lead"
assert pd.isna(lead_bands.iloc[5])
~~~
Validate the nonnegative/finite duration contract first. Otherwise a negative input outside the bins can quietly turn into missing. The example edges are illustrative, not established best H2 boundaries.

**Training-fitted bins:** quantile bins or other estimated edges must be learned inside fitting folds. `pd.qcut` is convenient for training exploration, but re-running it separately on validation gives a different meaning to the same category. Ties can also produce duplicate quantile edges.

Use a fitted discretizer such as KBinsDiscretizer or your own tested edge-preserving transformer when that route is justified. Define how new values outside the fitted range behave. Do not estimate bin edges from all rows or from targets merely to improve the final test.

## 17. Design a feature contract

### 17.1 A feature needs a specification

For each proposed feature record:

~~~text
name and version
business hypothesis
source fields
availability/prediction point
formula and units
missing/zero/invalid policy
whether fitting or labels are required
expected output type and names
task/model route
validation/ablation plan
~~~

If you cannot write this, you are not ready to put the feature into a production input path.

### 17.2 Candidate classes

- **Row-wise deterministic:** total nights, calendar features, explicit flags.
- **Distribution-fitted:** quantile caps, bins, frequency encodings.
- **Target-dependent:** target means, supervised selectors, LDA.
- **Time-dependent:** rolling histories, as-of aggregates, recency.

A feature can belong to several classes. For example, a smoothed historical cancellation rate is target-dependent and time-dependent.

### 17.3 Preserve names, rows, and interpretation

A feature builder should normally preserve the row count, index and order. Do not drop rows inside it while y remains unchanged.

Prefix/naming conventions can help, but names do not guarantee validity. Keep raw source names available in a mapping. Describe units in metadata, not only in a notebook comment.

Avoid accidental identifier features such as source ordinal, CSV index, run ID, or a split label. Do not include audit-only flags that reveal the final outcome or eligibility of a held-out row.

### 17.4 Crossed and polynomial features

Interactions can express hypotheses, such as meal by market segment or party size by stay duration. They can also create large sparse spaces and rare combinations.

Concatenating text with a delimiter can cause collisions if categories contain that delimiter. Use an unambiguous representation and missing policy. Do not blindly cross every pair of high-cardinality IDs.

PolynomialFeatures can add powers/interactions for a small selected numerical set. The feature count grows quickly. Put any fitted selection/scaling within CV and use regularization. A high-order polynomial is not intrinsically more informative.

**Checkpoint:** Design one feature, then identify a realistic record for which its naive implementation would be wrong.

## 18. Engineer H2 booking features

These are candidates to study, not an approved list. Every source field still needs the timing audit.

### 18.1 Stay features

| Feature | Formula / idea | Edge cases |
| --- | --- | --- |
| TotalNights | Weekday nights + weekend nights | Missing component stays unknown; zero-length stays |
| WeekendShare | Weekend nights / total nights | Zero denominator; missing numerator/denominator |
| HasWeekendStay | Weekend nights > 0 | Preserve unknown when nights are missing |
| ZeroNightStay | Explicit zero total nights | Audit whether valid; do not automatically delete |
| StayLengthBand | Domain-fixed or training-fitted bins | Record bin edges and boundary conventions |

Keeping TotalNights plus both components introduces redundancy. It may be useful for some models but also double-weight duration in clustering. Test the representation rather than assuming "more columns" is better.

### 18.2 Party features

| Feature | Formula / idea | Edge cases |
| --- | --- | --- |
| TotalGuests | Adults + Children + Babies | Any unknown component can make the exact total unknown |
| NonAdultGuests | Children + Babies | Missing must not silently become zero |
| HasChildrenOrBabies | Any known positive child/baby count | Known positive can establish true even if another count is missing |
| IsSoloParty | Exactly one total guest | Requires known total |
| NightsPerGuest | TotalNights / TotalGuests | Zero guest count, missing total; whether the ratio makes business sense |

Do not divide by guests merely because the numbers are available. Booking-level ADR is not automatically a per-person rate, and ADR-derived ratios are invalid ADR predictors.

A useful strict-total helper:

~~~python
def strict_total(frame, columns):
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Total needs distinct declared components.")
    selected = frame.loc[:, list(columns)]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes):
        raise TypeError("Parse numerical components before adding them.")
    # Unknown component means the exact total is unknown.
    return selected.sum(axis=1, min_count=len(columns))
~~~
A safe ratio helper for nonnegative counts/durations:

~~~python
def safe_nonnegative_ratio(numerator, denominator):
    numerator = pd.to_numeric(pd.Series(numerator, copy=False), errors="raise")
    denominator = pd.to_numeric(pd.Series(denominator, copy=False), errors="raise")
    numerator = pd.Series(
        numerator.to_numpy(dtype="float64", na_value=np.nan),
        index=numerator.index,
    )
    denominator = pd.Series(
        denominator.to_numpy(dtype="float64", na_value=np.nan),
        index=denominator.index,
    )
    if not numerator.index.equals(denominator.index):
        raise ValueError("Ratio inputs are misaligned.")
    if (numerator < 0).any() or (denominator < 0).any():
        raise ValueError("This ratio contract expects nonnegative inputs.")
    if np.isinf(numerator).any() or np.isinf(denominator).any():
        raise ValueError("Infinite ratio input.")
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = numerator.notna() & denominator.notna() & denominator.gt(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result
~~~
This contract returns missing for 0/0 or positive/0, not infinity or an invented zero. It intentionally does not define ratios involving negative values.

### 18.3 Calendar and lead-time features

Month cyclic pair:

~~~python
month_number = pd.Series([12, 1, 6], dtype="float64")
month_sin = np.sin(2 * np.pi * (month_number - 1) / 12)
month_cos = np.cos(2 * np.pi * (month_number - 1) / 12)
~~~
Both sine and cosine are needed to distinguish positions around the cycle. Month 12 and 1 become nearby. The origin is a convention; preserve it across training/inference.

Other candidates:

- arrival day-of-week sine/cosine with period seven;
- weekend-arrival flag with a declared business convention;
- LogLeadTime for validated nonnegative durations;
- predeclared lead-time bands versus training-fitted bins;
- complete arrival date for audit/split metadata;
- holiday/event indicators from an appropriate known calendar.

A guest's Country is not the hotel's location. A local demand-calendar feature should use the relevant location/calendar, not automatically the guest-origin calendar.

### 18.4 History features

TotalPreviousBookings = PreviousCancellations + PreviousBookingsNotCanceled.

A raw historical cancellation ratio is undefined when the total is zero. Keep a NoKnownHistory flag and an explicit missing/fallback policy. A Bayesian-style smoothed ratio can use:

~~~text
(previous cancellations + alpha) /
(total previous bookings + alpha + beta)
~~~

Positive alpha/beta encode a prior. Fixed domain priors and data-estimated/tuned priors are different choices. Any learned prior must respect training boundaries.

Most importantly, verify what the source counters actually know **as of your prediction point**. "Previous booking" can refer to creation order while outcomes become known later. Historical-looking columns are not automatically safe.

### 18.5 Requests, rooms, payment and identifiers

Potential features such as HasParkingRequest, SpecialRequestBand or room-type mismatch require availability review. A mismatch uses AssignedRoomType and inherits its timing risk.

HasAgent/HasCompany can represent known absence or missingness only if that interpretation is justified. Do not infer a travel agency's quality/rank from its numeric ID.

DepositType-derived features inherit its payment-derived timing. Turning it into a binary flag does not remove the source issue.

**Exercise:** Pick five candidates. For each, write a hypothesis, counterexample/edge case, availability explanation, and a development-only ablation plan.


## 19. Aggregates, histories, and target encoding

### 19.1 Distribution summaries versus historical events

A training-fitted category frequency is a learned distribution parameter available when the model is deployed. It is not automatically "how many bookings had happened before this particular booking."

A rolling history makes a stronger row-specific temporal claim. For that claim, each contributing observation and its outcome must have been available before the current prediction time.

Distinguish:

- source event time;
- source record/label resolution time;
- ingestion/publication time;
- prediction time;
- model training cutoff.

A record with an old event date can still contain a correction published later. Filtering by event time alone can introduce look-ahead.

### 19.2 Safe as-of joins

For external/history features, use a backward availability-based match when that is the declared question. Sort globally by the join time key and preserve a row ID so you can restore original order afterward.

Synthetic example:

~~~python
requests = pd.DataFrame(
    {
        "RowID": [0, 1, 2],
        "Group": ["A", "B", "A"],
        "prediction_at": pd.to_datetime(["2026-01-03", "2026-01-03", "2026-01-05"]),
    }
)
history = pd.DataFrame(
    {
        "Group": ["A", "B", "A"],
        "available_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"]),
        "KnownBookings": [2, 8, 9],
    }
)
joined = pd.merge_asof(
    requests.sort_values("prediction_at"),
    history.sort_values("available_at"),
    left_on="prediction_at",
    right_on="available_at",
    by="Group",
    direction="backward",
    allow_exact_matches=False,
    tolerance=pd.Timedelta("30D"),
).sort_values("RowID")
assert joined["KnownBookings"].tolist() == [2, 8, 9]
~~~
`direction="nearest"` could choose a future record. `allow_exact_matches=False` implements a strict earlier-than policy; use equality only if simultaneous availability is actually established. `tolerance` defines how stale a matched history is allowed to be.

Sorting by group first can violate merge_asof's globally sorted time-key requirement. Resolve time zones and event ordering explicitly when you have real timestamps. H2's date-only export does not magically supply this event stream.

### 19.3 Aggregating outcomes

A cancellation-rate feature requires both a category/history definition and outcome availability. If a previous booking's cancellation status was not yet known, it cannot contribute as resolved history at the current prediction time.

Do not calculate a group's ADR mean or cancellation rate on the complete CSV and merge it back into every row. That can include the row's own target, held-out targets, and future outcomes.

For a model-training representation, cross-fitted target encoding is one route. For a live historical-risk feature, past-only aggregation is a different route. Neither should be mislabeled as the other.

### 19.4 TargetEncoder and internal cross-fitting

A target encoder replaces a category with a smoothed estimate related to the target. For continuous ADR, explicitly use the continuous-target interpretation when appropriate. For binary cancellation, use the binary route.

In sklearn 1.9, use an explicit splitter to control shuffling/reproducibility. This example uses synthetic unique IDs to expose the difference between proper training encoding and a bad self-label representation:

~~~python
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

ids = pd.DataFrame({"Agent": [f"id-{i}" for i in range(40)]})
toy_y = np.array([0, 1] * 20)
crossfit = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_encoder = TargetEncoder(
    target_type="binary",
    smooth="auto",
    cv=crossfit,
)
training_encoding = target_encoder.fit_transform(ids, toy_y)

# Bad training representation for demonstration: each unique ID sees its own label.
self_label_encoding = target_encoder.fit(ids, toy_y).transform(ids)
assert not np.allclose(training_encoding, self_label_encoding)
~~~
Use fit_transform for training representations and transform for genuinely held-out/new observations. Within a Pipeline, fit_transform is used for intermediate fitting steps, preserving the encoder's internal cross-fitting route.

A small number of categories and smoothing do not remove the need for correct boundaries. If adding target encoding, compare it against a simpler one-hot baseline and retain the chosen encoder inside the saved pipeline.

### 19.5 Group/time limitations

The encoder's internal CV is inside the outer model-selection CV. Both layers must respect the intended dependence/availability rules.

In sklearn 1.9, TargetEncoder can accept CV objects/iterables and can route groups with the appropriate metadata-routing configuration. Verify the exact API and pass/request metadata deliberately.

Its cross-fitting split must place every sample in a validation fold exactly once. Ordinary TimeSeriesSplit leaves early observations without such a validation assignment, so it is not automatically a valid drop-in encoder CV scheme. A causal encoder may need a warm-up policy and custom expanding histories. Do not force future rows into early encodings just to satisfy a partition requirement.

Older sklearn versions differ in supported CV parameters; do not copy a 1.9 configuration into an older environment without checking it.

**Checkpoint:** With unique category IDs, explain why fitting target means and transforming those same rows can reveal the answer directly.

Sources: [merge_asof](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.merge_asof.html), [TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html).

## 20. Select and evaluate features

### 20.1 Availability comes before predictive power

First reject unavailable/invalid features. Then investigate usefulness among admissible candidates. Feature importance cannot justify a future variable.

Keep a small baseline representation. Add one coherent group of features at a time and use the same development protocol and fitting budget.

### 20.2 Selection routes

| Route | What it examines | Main caution |
| --- | --- | --- |
| Domain selection | Meaning, availability, reliability, cost | Record assumptions; do not disguise guesses as facts |
| Variance/missingness filters | Low observed variation or usable coverage | Fit thresholds on training; rare can be important |
| Correlation analysis | Redundancy/relationships | Pearson is linear; correlations do not prove causality or availability |
| Mutual information | Potential nonlinear target relationship | Estimation, discrete/continuous settings, cardinality and sampling matter |
| Embedded selection | Coefficients or model-driven selection | Estimator/scaling/regularization dependent |
| Recursive/sequential selection | Scores of feature subsets | Large repeated-fit cost and selection overfitting |
| Permutation importance | Score drop after perturbing a fitted input | Depends on model, metric, cohort and correlations |

Pearson and Spearman answer different questions. A correlation matrix on integer-coded countries is not a valid analysis of a nominal quantity.

Do not drop one of two correlated fields only because a heatmap is dark. Consider whether one is a deterministic duplicate, differently reliable, earlier-available, or useful through nonlinear interactions.

### 20.3 Importance is not an intrinsic property

A large coefficient depends on feature scale and coding. Tree impurity importances can favor high-cardinality inputs. A correlated substitute can make a useful field look unimportant under ordinary permutation.

Compute permutation importance on appropriate development scoring data while choices are still open. Permuting one raw variable can create unrealistic combinations with correlated fields. For engineered groups, consider group-wise ablation/permutation and report its meaning.

Do not examine final test importances to choose a smaller model and then reuse the same test as fresh evidence.

### 20.4 Leakage-safe selection placement

If a selector learns from X or y, put it inside the CV pipeline. Selecting the "best ten features" on the full development set before CV lets every CV scoring fold influence the subset.

Use nested evaluation or a separate development validation strategy when extensive selection/search would otherwise overfit the same evidence.

### 20.5 Ablation worksheet

For each change record:

~~~text
hypothesis | feature group added/removed | source availability
baseline recipe | candidate recipe | same split/cohort?
same estimator/search budget? | primary metric and variability
runtime/memory | stability | decision | limitations
~~~

Test groups such as stay/party features, cyclic dates, history features, missing indicators, and rare-category grouping separately before combining everything.

If a complex feature adds no stable benefit, removing it can improve clarity and robustness. Negative results are learning evidence, not failed work to hide.

**Checkpoint:** Explain why "feature importance is low" is insufficient on its own to justify deletion.

Sources: [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html), [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

## 21. Choose task-specific preparation routes

### 21.1 Regression

Start with approved inputs, explicit missingness, one-hot categories and a suitable numeric branch. For Ridge/linear methods, scaling affects regularization. For trees, numerical scaling is usually unnecessary, but category/missing handling remains model-specific.

If transforming ADR, preserve a valid inverse transformation and score in original units. Negative/zero ADR and eligibility decisions must be resolved first. An input-cleaning pipeline does not automatically solve target transformation.

### 21.2 Classification

Use the cancellation allowlist and preserve class meaning. ADR may be a transaction-derived outcome unavailable at the chosen prediction time; do not automatically include it because it is another column.

Scale appropriately for logistic regression, KNN and kernel SVM routes. Imbalance strategies and thresholds should be selected using valid development evidence, with any resampling inside fitting folds.

Do not let an imputation or feature-selection procedure see held-out class labels.

### 21.3 Clustering

Choose attributes that answer the segmentation question. Numeric-only, mixed one-hot, and specialized mixed-distance routes are alternatives, each requiring justification.

Distances can be dominated by:

- a large-unit duration;
- many one-hot columns from one source field;
- rare standardized binary indicators;
- duplicated totals/components;
- source IDs or outcome proxies;
- missing-value patterns.

Inspect the geometry, not only an internal score. Cluster profiles should be interpreted in original units. A two-dimensional PCA picture does not replace evaluating the actual feature space.

For full-H2 descriptive clustering, keep that workflow separate from supervised selection. If cluster labels/distances become predictors, the entire clustering preparation/fitting route belongs inside supervised training folds.

### 21.4 Representation methods

PCA learns directions from X; LDA uses y. Both require the right input form and fold-fitting boundaries. Centered dense PCA/LDA can be expensive after high-cardinality one-hot encoding.

A sparse alternative such as TruncatedSVD is useful in some routes, but is not a substitute for the brief's explicit PCA experiment. Record whether categories are reduced, retained separately, or omitted under a justified policy.

### 21.5 Three sensible starting recipes

| Route | Preparation starting point | Later alternatives |
| --- | --- | --- |
| Linear/kernel supervised | Fixed feature builder, numeric median/scaling, categorical sentinel/one-hot | Robust scale, frequency grouping, selected interactions, PCA/LDA |
| Tree supervised | Same valid feature policy, appropriate missing/category handling | Native categories where supported, no numeric scaling, monotonic transforms if useful |
| Distance clustering | Outcome-free attributes, explicit block weighting and scaling | Log/robust transforms, selected categories, alternate distance/representation |

These are starting hypotheses. Do not force a single preprocessing matrix onto every model merely to make the code shorter.


## 22. Build a complete synthetic pipeline

This worked route is intentionally small. It demonstrates mechanics and contracts, not an approved real-H2 feature policy or measured hotel performance.

### 22.1 Assemble your own module

Create `preparation.py` in your learning directory, or the equivalent module inside your own package. Put the helper definitions from Chapters 7, 8, 9 and 18 there, together with their imports:

- normalize_text_columns;
- parse_numeric_series;
- MONTH_NUMBER and make_arrival_dates;
- strict_total;
- safe_nonnegative_ratio;
- categorical_boundary if using it elsewhere.

Add the following fixed example contract and transformer. The source CSV still has 31 variables; this transformer receives only an explicit **predictor subset**, selected after the task availability review.

~~~python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

RAW_INPUT_COLUMNS = (
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateDayOfMonth",
    "MarketSegment",
    "Meal",
)
COUNT_INPUTS = (
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
)
NUMERIC_OUTPUTS = (
    "TotalNights",
    "TotalGuests",
    "WeekendShare",
    "LogLeadTime",
    "ArrivalMonthSin",
    "ArrivalMonthCos",
    "ChildrenMissing",
    "BabiesMissing",
    "HasChildrenOrBabies",
    "ZeroNightStay",
)
CATEGORICAL_OUTPUTS = ("MarketSegment", "Meal")


class BookingFeatureBuilder(TransformerMixin, BaseEstimator):
    def _build(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected named booking columns in a DataFrame.")
        if X.columns.duplicated().any() or set(X.columns) != set(RAW_INPUT_COLUMNS):
            raise ValueError("Raw input schema differs from this recipe's contract.")

        result = X.loc[:, list(RAW_INPUT_COLUMNS)].copy(deep=True)
        result = normalize_text_columns(
            result,
            columns=RAW_INPUT_COLUMNS,
            tokens_by_column={name: ["", "NULL"] for name in RAW_INPUT_COLUMNS},
        )
        for column in COUNT_INPUTS:
            result[column] = parse_numeric_series(
                result[column],
                integer=True,
                minimum=0,
            )
        dates = make_arrival_dates(result)

        result["TotalNights"] = strict_total(
            result,
            ["StaysInWeekendNights", "StaysInWeekNights"],
        )
        result["TotalGuests"] = strict_total(
            result,
            ["Adults", "Children", "Babies"],
        )
        result["WeekendShare"] = safe_nonnegative_ratio(
            result["StaysInWeekendNights"],
            result["TotalNights"],
        )
        lead = result["LeadTime"].to_numpy(dtype="float64", na_value=np.nan)
        result["LogLeadTime"] = np.log1p(lead)
        month = dates.dt.month.astype("float64")
        result["ArrivalMonthSin"] = np.sin(2 * np.pi * (month - 1) / 12)
        result["ArrivalMonthCos"] = np.cos(2 * np.pi * (month - 1) / 12)
        result["ChildrenMissing"] = result["Children"].isna().astype("float64")
        result["BabiesMissing"] = result["Babies"].isna().astype("float64")

        known_positive = result["Children"].gt(0).fillna(False) | result["Babies"].gt(0).fillna(
            False
        )
        known_zero = (result["Children"].eq(0) & result["Babies"].eq(0)).fillna(False)
        family = pd.Series(np.nan, index=result.index, dtype="float64")
        family.loc[known_positive] = 1.0
        family.loc[known_zero] = 0.0
        result["HasChildrenOrBabies"] = family
        result["ZeroNightStay"] = result["TotalNights"].eq(0).astype("Float64")

        for column in NUMERIC_OUTPUTS:
            result[column] = result[column].to_numpy(
                dtype="float64",
                na_value=np.nan,
            )
        for column in CATEGORICAL_OUTPUTS:
            if result[column].dropna().eq("__MISSING__").any():
                raise ValueError("Raw category collides with the reserved missing token.")
            result[column] = (
                result[column]
                .astype(object)
                .where(
                    result[column].notna(),
                    np.nan,
                )
            )
        return result

    def fit(self, X, y=None):
        prepared = self._build(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.asarray(prepared.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        return self._build(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("Unexpected input feature names.")
        return self.feature_names_out_.copy()
~~~
Notice:

- Extra/missing predictor columns, including accidental targets, are rejected.
- Raw field order can be restored by name; row order/index remain unchanged.
- Count parsing rejects invalid strings/fractions/negative counts.
- Calendar validation rejects incomplete/impossible dates under this example contract.
- No imputation statistics or category vocabulary are learned in the feature builder.
- Unknown party totals stay unknown before fitted imputation.
- A known positive child count establishes family=true even if Babies is missing.
- All numeric output types and category missing representations are explicit.
- The full raw subset remains in the intermediate DataFrame, but the next ColumnTransformer selects only declared model features.

This fixed feature policy is pedagogic. For real H2, review every input's timing and decide which derived features are admissible for the task.

### 22.2 Add fitted preparation and an estimator

In the same module:

~~~python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_booking_pipeline(alpha=1.0):
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
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    columns = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_OUTPUTS)),
            ("categorical", categorical, list(CATEGORICAL_OUTPUTS)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        [
            ("features", BookingFeatureBuilder()),
            ("prepare", columns),
            ("model", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )
~~~
The first step constructs/validates features. The second learns medians, scale parameters and category vocabulary. The final step learns regression coefficients. A CV clone refits the whole chain using only that fold's fitting rows.

The example derives totals before imputation so unknown source components remain explicit. Independently imputing derived variables can break raw arithmetic identities in transformed space. An alternative is to impute base quantities first and then derive totals, which preserves arithmetic but constructs totals from invented values. State which meaning you chose; do not call either result an observed fact.

To test a clipping route, factor deterministic parsing, clipping and dependent feature construction into the required order. Do not blindly insert a clipper before raw strings have been normalized or after an unclipped duplicate feature already carries the same information.

### 22.3 Run it on synthetic bookings

Create a separate learning script that imports the module you just wrote. The following source creates valid synthetic calendar/booking fields and deliberately missing Children values:

~~~python
import numpy as np
import pandas as pd
from preparation import make_booking_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

rng = np.random.default_rng(42)
n = 48
lead = rng.integers(0, 180, size=n)
weekend = rng.integers(0, 3, size=n)
weekday = rng.integers(0, 6, size=n)
weekend[0] = 0
weekday[0] = 0
children = rng.integers(0, 3, size=n).astype(float)
children[::7] = np.nan

X = pd.DataFrame(
    {
        "LeadTime": lead,
        "StaysInWeekendNights": weekend,
        "StaysInWeekNights": weekday,
        "Adults": rng.integers(1, 4, size=n),
        "Children": children,
        "Babies": rng.integers(0, 2, size=n),
        "ArrivalDateYear": [2016] * n,
        "ArrivalDateMonth": ["June"] * n,
        "ArrivalDateDayOfMonth": [(i % 28) + 1 for i in range(n)],
        "MarketSegment": ["Direct", "Online", "Agency"] * 16,
        "Meal": ["BB", "HB", "SC"] * 16,
    },
    index=np.arange(100, 100 + n),
)
y = pd.Series(
    60 + 0.2 * lead + 3 * (weekend + weekday) + rng.normal(0, 3, size=n),
    index=X.index,
    name="ADR",
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)
pipeline = make_booking_pipeline()
cv = KFold(n_splits=3, shuffle=True, random_state=42)
cv_mae = -cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error",
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_valid)
assert np.isfinite(predictions).all()
print("Synthetic CV MAE:", cv_mae.mean())
print("Synthetic validation MAE:", mean_absolute_error(y_valid, predictions))

new_booking = X_valid.head(1).copy()
new_booking["MarketSegment"] = "New market"
assert np.isfinite(pipeline.predict(new_booking)).all()
~~~
Synthetic y was generated for teaching. A successful score here is not evidence that the real H2 feature policy is valid.

### 22.4 Inspect the fitted representation

~~~python
prepared_frame = pipeline.named_steps["features"].transform(X_train)
matrix = pipeline.named_steps["prepare"].transform(prepared_frame)
names = pipeline.named_steps["prepare"].get_feature_names_out()
assert matrix.shape[0] == len(X_train)
assert matrix.shape[1] == len(names)
print(matrix.shape)
print(names)
~~~
A sparse matrix might become dense if the combined density exceeds the declared threshold. Inspect actual type/dtype/shape rather than assuming it from one encoder argument.

If you add feature selection, also store its support mask/output names. Coefficients must be interpreted against the exact final ordered feature list.

### 22.5 Persist and reload

~~~python
from pathlib import Path

import joblib

destination = Path("artifacts/learning-only")
destination.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, destination / "booking_pipeline.joblib")
restored = joblib.load(destination / "booking_pipeline.joblib")
np.testing.assert_allclose(
    pipeline.predict(X_valid),
    restored.predict(X_valid),
)
~~~
Use unique run directories for real experiments; this fixed learning path is not your scientific versioning strategy. Keep artifacts ignored, record environment and input contract, and only load trusted files.

The importable module containing custom helpers/classes must exist when loading in a new process. Do not rely on an object defined only in a vanished Colab cell.

**Checkpoint:** Explain the raw-to-feature-to-matrix-to-prediction path for one record. Point to the exact objects learning medians, vocabulary, scale and coefficients.


## 23. Test preparation and inference

### 23.1 What tests can and cannot prove

Tests can prove that your implementation follows a declared rule on checked cases. They cannot prove that an exported field was genuinely available at booking creation. That needs source/domain evidence.

Test meaningful failure modes rather than every line of a pandas chain.

| Area | Cases to include |
| --- | --- |
| CSV/header | BOM, quoted commas, duplicate headers, normalization collision, unexpected columns |
| Token normalization | Whitespace, literal NULL versus meaningful category, leading-zero ID |
| Numeric parsing | Blank, missing, invalid text, fraction, infinity, negative count, binary label 2 |
| Dates | Leap day, invalid month/day combination, missing component, source week convention |
| Missingness | Unknown child, all-missing fitting column, newly missing held-out field |
| Duplicates/splits | Original IDs/order preserved; group/cohort rules and manifest alignment |
| Outliers | Bounds learned only in fitting; extreme held-out value does not change them |
| Features | Zero denominator, partial missing total, positive-known family indicator, nonmutation |
| Encoding | Unknown and rare categories, reserved token collision, stable width/order |
| Cross-fitting | Target-dependent training representation excludes the row's own outcome |
| Pipeline | Fresh cloned fit per fold, finite valid output, correct feature names |
| Persistence | Same raw inputs give same predictions after reload in a fresh process |

### 23.2 Focused unittest examples

After implementing the helpers, place this in `tests/test_preparation.py`. The optional clipper test assumes you also wrote Chapter 12's transformer.

~~~python
import unittest

import numpy as np
import pandas as pd
from preparation import (
    TrainingQuantileClipper,
    make_arrival_dates,
    normalize_text_columns,
    parse_numeric_series,
    safe_nonnegative_ratio,
    strict_total,
)


class PreparationTests(unittest.TestCase):
    def test_identifier_padding_does_not_destroy_leading_zeros(self):
        raw = pd.DataFrame({"Agent": [" 0012 ", "NULL"]})
        clean = normalize_text_columns(raw, ["Agent"], {"Agent": ["NULL"]})
        self.assertEqual(clean.loc[0, "Agent"], "0012")
        self.assertTrue(pd.isna(clean.loc[1, "Agent"]))
        self.assertEqual(raw.loc[0, "Agent"], " 0012 ")

    def test_fractional_and_nonfinite_counts_are_rejected(self):
        for value in ["1.5", "inf", "-1", "two"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_numeric_series(
                        pd.Series([value], name="Adults"),
                        integer=True,
                        minimum=0,
                    )

    def test_partial_missing_total_stays_unknown(self):
        raw = pd.DataFrame(
            {
                "Adults": [2.0],
                "Children": [np.nan],
                "Babies": [0.0],
            }
        )
        total = strict_total(raw, ["Adults", "Children", "Babies"])
        self.assertTrue(pd.isna(total.iloc[0]))

    def test_zero_denominator_does_not_create_infinity(self):
        ratio = safe_nonnegative_ratio(
            pd.Series([1.0, 0.0, np.nan]),
            pd.Series([0.0, 0.0, 2.0]),
        )
        self.assertTrue(ratio.isna().all())

    def test_non_leap_february_29_is_rejected(self):
        raw = pd.DataFrame(
            {
                "ArrivalDateYear": [2015],
                "ArrivalDateMonth": ["February"],
                "ArrivalDateDayOfMonth": [29],
            }
        )
        with self.assertRaises(ValueError):
            make_arrival_dates(raw)

    def test_holdout_does_not_change_fitted_clipping_bounds(self):
        training = pd.DataFrame({"LeadTime": [0.0, 10.0, 20.0]})
        clipper = TrainingQuantileClipper(
            columns=("LeadTime",),
            lower=0.2,
            upper=0.8,
        ).fit(training)
        upper_before = clipper.upper_bounds_.copy()
        transformed = clipper.transform(pd.DataFrame({"LeadTime": [100000.0]}))
        self.assertEqual(transformed.loc[0, "LeadTime"], upper_before["LeadTime"])
        pd.testing.assert_series_equal(upper_before, clipper.upper_bounds_)


if __name__ == "__main__":
    unittest.main()
~~~
Run with the intended interpreter:

~~~powershell
python -m unittest discover -s tests -v
~~~

If using a packaged module, change imports consistently to your module path. Do not make tests pass by secretly importing code from the old repository.

### 23.3 Inspect fitted fold state

Using the synthetic training data/pipeline from Chapter 22:

~~~python
from sklearn.model_selection import cross_validate

folds = list(cv.split(X_train, y_train))
fold_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=folds,
    scoring="neg_mean_absolute_error",
    return_estimator=True,
)
for (fitting_positions, _), fitted in zip(folds, fold_results["estimator"]):
    fitting_features = fitted.named_steps["features"].transform(X_train.iloc[fitting_positions])
    expected = fitting_features["TotalGuests"].median()
    numeric_imputer = (
        fitted.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
    )
    column_position = list(NUMERIC_OUTPUTS).index("TotalGuests")
    np.testing.assert_allclose(
        numeric_imputer.statistics_[column_position],
        expected,
    )
~~~
Import `NUMERIC_OUTPUTS` from your own preparation module for this inspection. The example inspects a specific fitted statistic; it is not a proof that every aspect of the scientific protocol is valid.

### 23.4 Fresh-process and serving tests

- Load your pipeline in a fresh interpreter with the same module/environment.
- Predict an input with shuffled raw column order if your contract supports name-based reordering.
- Reject extra target columns and absent required columns.
- Test a valid unknown category, a literal reserved missing token, a negative count and an impossible date.
- Ensure predictions do not call fit or rewrite learned parameters.
- Verify both batch and single-record paths construct the same raw schema.
- Check that output rows align with input IDs even after errors/quarantine handling.

Do not refit an imputer/scaler on each prediction batch. That makes a booking's prediction depend on which other bookings happened to be in the same request.

**Checkpoint:** Demonstrate at least one deliberately failing test, then explain the fix and why the test would catch a recurrence.

## 24. Version decisions and outputs

### 24.1 Keep a preparation manifest

Record:

~~~text
source hash and parser options
column-name mapping
token and type rules
row eligibility/quarantine rules and counts
source row IDs and split/cohort identity
prediction point and task allowlist
feature formulas, units, edge-case policies
fitted imputation/encoding/scaling/cap/selection state
raw and transformed column names/order
library versions, source SHA and recipe version
validation/ablation evidence and limitations
~~~

A config name alone is not enough. If a recipe references a changed lookup table or a different source file, it is a different effective experiment.

### 24.2 Raw, canonical, and model-ready are different artifacts

- **Raw:** original immutable bytes.
- **Canonical:** declared deterministic parsing/normalization, with traceable source IDs.
- **Training representation:** the output of the current fitted/fold-specific preparation.
- **Inference representation:** transform using the chosen saved pipeline.

TargetEncoder's out-of-fold training representation is not generally recreated by calling transform on its training rows. Record the distinction when caching/exporting.

### 24.3 Storage alternatives

| Format | Useful for | Caution |
| --- | --- | --- |
| CSV | Human-readable normalized snapshots and reports | Types, missing tokens, leading zeros and float/date round trips need explicit contracts |
| Parquet | Typed analytical tables | Usually needs an engine such as PyArrow; record versions and schema |
| Sparse NPZ | Sparse numerical matrices | Also save ordered feature names, row IDs and transformer identity |
| joblib | Your trusted fitted pipeline | Load only trusted objects in compatible environments |
| JSON | Manifests, policies, metrics, mappings | Explicit finite/undefined-value policy; large arrays belong elsewhere |

A deterministic canonical snapshot can be useful, but it does not replace fitting preparation inside CV. Keep generated datasets/matrices/artifacts out of normal source commits.

If choosing Parquet, install the selected engine deliberately, for example `python -m pip install pyarrow`. A dtype preserved in a table file does not automatically preserve the intended categorical vocabulary/model handling.

Do not persist pandas `.cat.codes` as if it were a universal category mapping. Codes can differ between independently constructed batches; save and apply the actual encoding contract.

### 24.4 Version when scientific meaning changes

A new feature set, missingness interpretation, clipping policy, representation, lookup, eligibility rule or prediction point deserves a new identified recipe/policy. Preserve configs once used in reported comparisons.

Re-running an unchanged recipe creates a new run ID. Changing model parameters does not require copying your entire source tree. The source commit and saved manifest connect the recipe to the implementation.

### 24.5 Monitor new inputs without silently adapting

Useful diagnostics include missing rates, parse failures, unseen category counts, changed ranges and changes in input distributions.

A shift alert is not permission to fit new preprocessing on live requests. Investigate and explicitly retrain/version on an appropriate labeled dataset when justified. Feature-drift tests are not proof of performance drift, and an unchanged input distribution does not prove unchanged outcome relationships.

## 25. Troubleshooting and bad examples

### 25.1 Common implementation failures

| Symptom | Likely issue | First check |
| --- | --- | --- |
| Missing counts unexpectedly zero | Literal strings were not mapped | Inspect raw token frequencies and declared token rules |
| IDs lose zeros | Numeric inference/conversion | Read IDs literally and review the mapping |
| pd.NA triggers an estimator error | Incompatible boundary/dtype | Convert selected inputs to the expected np.nan/numeric form consistently |
| Fractions silently become counts | Unvalidated integer cast | Check integrality before casting |
| February 31 passes | Only component ranges checked | Construct and validate the complete date |
| New category breaks prediction | Vocabulary/unknown policy absent | Inspect the fitted encoder, not current batch categories |
| Output width changes | Dynamic feature/indicator construction | Use fixed named output contracts |
| A ratio contains infinity | Zero denominator | Define an explicit missing/flag policy |
| Guest total seems too small | Default summation skipped unknowns | Use a strict total or document impute-first semantics |
| Cross-validation looks perfect | Outcome, target means or source-order leakage | Audit every input, derived feature and fitting boundary |
| Clipping erases important cases | Arbitrary/global thresholds | Compare valid alternatives on development data and inspect affected counts |
| Memory crash | Dense expansion or too many parallel copies | Inspect shape/density/dtype and estimate bytes before conversion |
| Clusters reflect only one category | Implicit encoding/block weights | Examine representation geometry and feature redundancy |
| Saved model cannot load | Missing custom module or version mismatch | Restore trusted code/environment and test fresh-process reload |
| Same booking changes across batches | Preparation refitted per batch | Apply the stored transform without fitting |
| TargetEncoder rejects a time splitter | Its validation partitions do not cover each row exactly once | Design a genuinely causal scheme rather than forcing future information in |

### 25.2 Bad patterns and their corrections

**Bad: global imputation before splitting.**

~~~python
# Wrong for a supervised evaluation:
# full_table["LeadTime"] = full_table["LeadTime"].fillna(full_table["LeadTime"].median())
# Then split and evaluate as if holdouts were untouched.
~~~
Correct: imputer inside the whole pipeline, fitted separately per fitting partition/fold.

**Bad: converting all missing values to zero.**

A missing Children count, missing company identifier, invalid date and unknown target are four different situations. Use explicit per-column contracts.

**Bad: making labels out of source correlations.**

Do not rewrite IsCanceled based on a status/price/booking pattern just because it agrees often. Audit discrepancies using source definitions.

**Bad: selecting features with final-test importance.**

Use development data and nested/fold-safe selection. Preserve final tests for the frozen procedure.

**Bad: outcome-derived segmentation.**

Removing IsCanceled after computing a category cancellation-rate feature does not make the resulting clustering outcome-free.

**Bad: parsing metadata as predictors.**

Source row number, source hash, split name and final-status date belong in metadata/audits, not automatically in X.

### 25.3 Learn from a failed transformation

When a transformation fails:

1. Reduce to a few synthetic rows reproducing it.
2. Identify the assumed schema/type/availability.
3. Distinguish invalid input from unsupported but legitimate data.
4. Decide the intended policy before patching.
5. Write the regression test.
6. Fix the shared transformation path.
7. Recheck feature names, row alignment, saved-state behavior and relevant development comparisons.
8. Document the policy/version change.

Do not fix a single notebook cell while leaving a different behavior in inference.

## 26. A practical learning sequence

Work in small verified pieces; create modules/folders only when real code/output needs them.

| Exercise | Implement | Deliverable / understanding check |
| --- | --- | --- |
| 1. Literal CSV | Read a dirty synthetic file | Explain preserved tokens, quotes, encoding and leading zeros |
| 2. Schema | Raw/canonical header validation | Detect duplicate/colliding/missing/extra names |
| 3. Types | Count, measurement, binary and ID handling | Reject fraction/infinity without confusing them with missingness |
| 4. Calendar | Reconstruct arrival dates | Leap-year and invalid-date tests |
| 5. Missingness | Compare semantic policies | Show effect on totals/flags; no invented zero labels |
| 6. Identity | Stable source IDs and duplicate audit | Distinguish bookings from feature patterns/customers |
| 7. Splits | Frozen row/group/time protocol | Prove disjointness/alignment and claimed availability constraints |
| 8. Learned preparation | Imputer, encoder, scaler, optional caps | Inspect learned statistics and holdout stability |
| 9. Row-wise features | Stay, party, calendar and history candidates | Document formula, units, missing/zero/timing policy |
| 10. Supervised encoding | Unique-ID target-encoding demonstration | Explain why fit_transform differs from fit().transform() |
| 11. Pipeline | Complete synthetic example | Run CV, unknown-category prediction, name inspection and reload |
| 12. Real-source audit | Apply reviewed checks to supplied H2 | Report actual findings without assuming synthetic patterns |
| 13. Ablation | Compare a small feature group | Same split/model budget and honest uncertainty |
| 14. Report | Preparation and leakage sections | Another person can reproduce every decision |

For Milestone 1, prioritize a trustworthy simple preparation route and one model per task over dozens of speculative features. For later work, extend only where a scientific hypothesis and development evidence justify it.

### Retrieval questions

Before using the real data, answer:

1. Which transformations are source-fixed, fitted from X, fitted from y, or time-dependent?
2. What is the latest availability of each feature's source fields?
3. What is the difference between invalid, missing, not applicable, rare and unseen?
4. Which rows and labels contributed to every fitted statistic?
5. Does every derived total/ratio preserve the intended missingness meaning?
6. How are groups, source IDs, y and weights kept aligned?
7. What happens to a legitimate new category?
8. Which artifacts are sufficient for identical inference after restart?
9. Which assumptions cannot be established from this snapshot dataset?
10. What evidence would make you reject a feature even if its validation score improves?

## 27. Final checklists and decision sheets

### 27.1 Cleaning/preparation checklist

- [ ] Raw source immutable and hashed; source labels/mapping documented.
- [ ] H2-only scope and expected schema/size verified against the actual file.
- [ ] Token, encoding, numeric, date and semantic type policies explicit.
- [ ] Invalid parses distinguished from genuine missingness.
- [ ] Missing targets never imputed; row eligibility/population declared.
- [ ] Duplicate/identity/dependence policy justified.
- [ ] Unusual values investigated rather than blindly removed.
- [ ] Original row IDs/order preserved through all aligned arrays.
- [ ] Split/cohort manifests frozen before adaptive preparation decisions.
- [ ] All learned preparation fitted within the correct training/fold boundary.
- [ ] Sparse/dense dimensions and memory budget inspected.

### 27.2 Feature-engineering checklist

- [ ] Per-task allowlists match the prediction/analysis point.
- [ ] Every feature has source, formula, units, type and availability.
- [ ] Zero denominators, partial missing totals and invalid inputs handled deliberately.
- [ ] Categories/unknowns/rare groups have fitted stable behavior.
- [ ] Historical/aggregate inputs respect actual information availability.
- [ ] Target-dependent features use appropriate cross-fitting/evaluation.
- [ ] Feature-selection and representation steps remain inside fitting boundaries.
- [ ] Redundancy/block weighting and estimator requirements considered.
- [ ] Ablations compare like-for-like protocols and budgets.
- [ ] Clustering stays outcome-free under its declared objective.

### 27.3 Reproduction/inference checklist

- [ ] Raw and transformed schemas/names/order are saved.
- [ ] Complete fitted pipeline, custom code, policy and environment recorded.
- [ ] No silent refitting or dynamic output-width changes at inference.
- [ ] Fresh-process reload and malformed/new-input tests pass.
- [ ] Generated artifacts are traceable, ignored by Git, and available when needed.
- [ ] Limitations and actual findings are reported honestly.

### 27.4 One-page decision worksheet

For each real issue or proposed feature, fill in:

~~~text
Issue/feature:
Task and prediction point:
Observed source evidence:
Possible interpretations:
Chosen rule and alternatives rejected:
Deterministic, fitted-X, fitted-y, or temporal?
Which fitting rows/labels/history may contribute?
Missing/zero/invalid/unseen behavior:
Effect on rows, units, names and dimensions:
Test cases:
Development comparison and budget:
Run/source/policy versions:
Residual limitation:
~~~

A completed worksheet is more valuable than a long cleaning function whose assumptions nobody can explain.

## 28. References and validation notes

### Primary references

- [Course project brief](../reference/CSE382_Project_Description_Milestones.pdf): original source labels, preparation requirements and leakage assessment. Preserve the `guides/` and `reference/` layout or repair this link when copying the handbook; see the [documentation index](../README.md).
- [Hotel booking demand datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/): original publication and source-variable construction.
- [pandas 2.3.3 read_csv](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.read_csv.html): literal parsing, dtype and missing-marker controls.
- [pandas 2.3.3 missing data](https://pandas.pydata.org/pandas-docs/version/2.3.3/user_guide/missing_data.html): missing representations and nullable types.
- [pandas 2.3.3 to_numeric](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.to_numeric.html): conversion and coercion behavior.
- [pandas 2.3.3 to_datetime](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.to_datetime.html): date parsing and invalid values.
- [pandas 2.3.3 merge_asof](https://pandas.pydata.org/pandas-docs/version/2.3.3/reference/api/pandas.merge_asof.html): availability-based joining mechanics.
- [sklearn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): consistent preparation and leakage boundaries.
- [sklearn imputation](https://scikit-learn.org/stable/modules/impute.html): simple, nearest-neighbor and iterative approaches.
- [sklearn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html): scaling, categorical encoding and nonlinear transformations.
- [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html): unknown/rare categories and output representation.
- [TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html): internal cross-fitting and version-specific CV support.
- [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html): Yeo-Johnson/Box-Cox domains and fitted behavior.
- [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html): fitted filters and model-driven selectors.
- [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html): evaluation-dependent feature inspection.
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html): trusted loading, custom modules and compatible environments.

### Example scope

The synthetic examples illustrate implementation behavior, not observed H2 distributions or a guaranteed valid real-H2 input policy. The future source audit and prediction-point decisions remain your responsibility.

The guide uses pandas 2.3.3 and sklearn 1.9.1 API behavior as its checked baseline. In particular, TargetEncoder CV splitter support differs from older versions. Inspect installed versions and matching official documentation before changing an environment.

### Validation record for the learning examples

The reference checks ran on Python 3.14, pandas 2.3.3, NumPy 2.5.4 and scikit-learn 1.9.1. They used synthetic CSVs/bookings only. They covered literal parsing and header collisions, token normalization, strict numeric/calendar validation, missing totals, nullable ratios, quantile bounds, sklearn cloning, missing indicators, one-hot unknown/rare behavior, TargetEncoder cross-fitting and time-split incompatibility, backward as-of joining, the full three-fold preparation pipeline, input-schema failures, feature/index preservation, and saved-pipeline predictions in a fresh Python process. The training-only quality plot and numerical profile also ran.

All six documented unittest cases passed. All 32 Python blocks passed syntax parsing, both PowerShell blocks passed syntax checks, and all 28 chapter-navigation links and two local companion/reference links resolved. Power-transform domain behavior and the bin-boundary example were also checked.

These checks are not an audit of H2, an approved feature list, evidence of hotel prediction performance, or proof that every optional method fits your chosen question. No real H2 data or paid Colab compute was used. Your manually written implementation still needs its own tests and availability review.
