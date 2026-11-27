# Synthetic matched-study verification

This is a synthetic implementation check. Its scores do not measure bike-share performance.

The search uses 45 base CV fits and three stack regularization candidates. B/H/C training roles separate base selection, stack selection and cutoff calibration. Seeds 42/43/44 vary estimator initialization only; the stack shares its seed-42 OOF construction. Different OOF base fitting sizes are disclosed. The test was already examined in prior work, so this is an extension of an existing benchmark. Fit timings exclude preprocessing and IO and do not establish a controlled speedup.

See supplemental/matched/README.md for the complete selection and reproduction method. Raw data, row IDs, predictions, models and machine logs are private.
