# Generated artifacts

Ignored execution outputs live at:

```text
artifacts/<task>/<version>/<UTC-timestamp>-<unique-id>/
  config.json
  manifest.json
  # Later training phases add:
  # metrics.json, search-results.csv, pipeline.joblib, figures/
```

A new run always gets its own directory. Do not overwrite earlier runs.
The bootstrap creates metadata only and reports status `initialized`.
Use `HOTEL_ML_ARTIFACT_ROOT` or `--output-root` to write directly to persistent storage in Colab.
Keep fitted artifacts locally for the live demonstration; exclude them from the submitted ZIP.
When reviewed results exist, create the relevant `reports/` folders and publish selected tables
and figures there. See [future structure](../docs/project/future-structure.md).
