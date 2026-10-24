# User-supplied dataset

Place the original **H2.csv** at `data/raw/H2.csv`. No dataset is downloaded automatically.
Expected original source: City Hotel only, 79,330 rows, 31 columns. Verify this against the actual file before training.

- `raw/`: immutable original files, ignored by Git.
- `interim/`: ignored temporary transformations and split manifests.
- `processed/`: ignored derived inputs.

Do not combine H1 and H2. A combined snake_case mirror is not a drop-in replacement:
isolate City Hotel and document an explicit one-to-one column mapping before using it.
Whitespace, string `NULL`, identifier columns, and source encoding need inspection.
Do not turn absent `Agent`/`Company` values into arbitrary numeric quantities.

The [original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) defines the variables.
A dataset hash identifies exact input bytes; it does not verify schema, quality, or feature availability.
