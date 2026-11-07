# Public artifact rules

This repository contains source code and aggregate experiment evidence. It does not publish personal names, contact information, user-profile paths, credentials, raw rider postal codes, row-level trips, trip membership lists, prediction rows, model binaries, or original document metadata.

The source code describes schema fields and a derived locality rule needed to explain the experiment. Field names and geographic prefix rules are not individual records. The actual trip-to-locality mapping stays local and is ignored by Git. Restoring private data for execution does not authorize committing those files.

Published JSON contains aggregate metrics, model parameters, candidate grids, completion metadata, input checksums, and audit attestations. Historical non-finite optional numeric values are normalized to JSON null; measured scores are not invented to replace undefined values. The export inventory records this normalization.

The export was checked for personal names and machine-specific paths before publication. `tools/check_public_artifacts.py` and CI enforce generic rules against contact addresses, profile paths, common credential formats, row-shaped JSON records, and prohibited data/binary file types. These automated checks complement the source review and intentionally conservative file selection.

Git commits use a generic project identity. GitHub still associates repository activity and pull requests with the authenticated account as part of its normal access controls.

Keep locally restored data under `data/`, model outputs under `results/`, and logs under `logs/`. The `.gitignore` file excludes these private artifacts. Aggregate publication belongs under `experiments/` or `validation/` and must pass the public-artifact checks.
