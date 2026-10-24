# Presentation layer

The reusable CLI lives in `src/hotel_ml/cli.py`.
A future Streamlit interface can call inference and evaluation modules without reimplementing ML logic.
The final application must expose dataset analysis, model comparison, training/tuning access,
saved-artifact regression/classification predictions, and clustering results.
