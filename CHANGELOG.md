# Changelog

## Unreleased

- Research structure option `peak-anchored`: keeps the most intense audio sections at the Planner density and caps the other sections at a human peak-relative ratio for the target star, calibrated from the local reference profile (`peak-anchored-density-v2`).
- Research generator star condition `section` (`section-local-star-v1`): non-peak sections receive the star implied by their density relative to the peak, using a calibrated star–peak-density slope.
- `difficulty-placement.json` for every run (`difficulty-placement-v2`): where the exported chart's highest strain peaks fall relative to peak sections, and planned versus exported section contrast. Stream bars are excluded until its generator bug is fixed.
- Opt-in Planner training losses `density_quantile` and `section_contrast` (audio `section_ids`); the default objective is unchanged. A seven-arm candidate study is recorded in docs/peak-allocation.md; no planner revision is promoted.

## 0.1.0

- Portable Python package and a single local workbench on port 8788.
- Four family-named generator revisions and the latest hierarchical planner with baseline comparisons.
- Verified, separately distributed inference assets without training-state or machine metadata.
- Platform data directories, explicit asset installation, offline CPU generation and runtime diagnostics.
- Generation reservation fixes for overlapping requests, exclusive Windows port binding, portable tests and release checks.
