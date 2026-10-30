# Local and Google Colab execution

One package and one set of recipes run in both environments.

1. Open `notebooks/colab_bootstrap.ipynb` in Colab after publishing.
2. Set `REVISION` to a commit SHA for a reproducible run; `main` is only the setup default.
3. Run clone/install cells and inspect `doctor`.
4. Put your H2.csv in `MyDrive/hotel-booking-intelligence/data/H2.csv`.
5. Run the optional Drive cell. It copies data locally for reads and sets the output root to Drive.
6. Initialize a run. Training cells will follow when the package implements training.

The notebook creates metadata only. It does not train or report fabricated scores.
Use the same code revision and recipe locally.

Colab VMs are temporary: `/content` files and in-memory objects may disappear.
Persist artifacts in Drive before disconnecting. The CLI honors `HOTEL_ML_ARTIFACT_ROOT`;
pass `--data` to record an input fingerprint. Copy CSVs from Drive once to avoid slow repeated I/O.

The preferred baseline is Python 3.12; supported runtimes are 3.12–3.14.
The notebook installs this repository's dependencies and records actual runtime versions.
A common resolved lock still needs validation before reported experiments.

Start with a **CPU** runtime: GPU selection does not accelerate scikit-learn automatically.
Free Colab guarantees neither a particular GPU/RAM allocation nor uninterrupted long jobs.
See the [official FAQ](https://research.google.com/colaboratory/faq.html).
Use VS Code for local development and the Colab notebook interface for remote execution.

Commit code and cleaned notebooks from a development checkout.
Do not publish notebook outputs containing data rows, Drive paths, credentials or fitted binaries.
If code changes during a Colab session, the manifest dirty flag matters; commit/review before
using the run as reproducible evidence.
