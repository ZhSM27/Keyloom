# Changelog

## 0.2.0

- Chart evaluator `keyloom evaluate` (`chart-evaluation-v4`): community pattern rules including Tech, within-style and whole-chart human likeness, hand pressure and accent alignment, calibrated against nearest human charts.
- Playtest candidate generator `music-accent` r1 (`accent-count-v1`, `accent-decoding-v1`): audio support decides demoted chords and chord sizes, Chordjacks keep their rows and follow a human size prior, Chordjacks and Gluts have shorter same-column chains, and bar motif reuse gives layouts a phrase memory. Drops, demotions and size changes are recorded per generation in `accent_support`. Its weights ship in `keyloom-models-r2.zip`; the catalog marks them optional, so `keyloom-models-r1.zip` still installs.
- Asset bundles may omit catalog files marked `optional`; `doctor` reports them without failing.
- The workbench preselects `music-accent`, `peak-anchored` and the section star condition so playtests use the newest options. This is not a promotion: the command line and catalog default stays `music-base`, and baseline options remain selectable.
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
