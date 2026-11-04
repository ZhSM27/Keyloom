# Peak-anchored section density

`structure_id = "peak-anchored"` (CLI `--structure peak-anchored`) is a research option. `legacy` and `hierarchical` are unchanged and remain the baselines.

## Why

osu!mania star rating is dominated by a chart's hardest stretch, not its average. The base Planner receives one song-wide star value for every bar and is trained with a log-space regression plus smoothness terms, so it tends to give verses roughly the climax density. The `hierarchical` option can only move density by ±12% and preserves the total note budget, which cannot create verse/chorus contrast.

## What it does

1. Build the same audio section/phrase hierarchy as `hierarchical` (`section-phrase-hierarchy-v1`).
2. Score each section by song-relative audio energy, the Planner's own log density (z-scored with a 0.1 spread floor, so near-equal densities do not create rank) and whether the section returns.
3. Take sections from the most intense down until they cover at least 30% of active bars: these are the **peak** (`peak-anchored-density-v2`). A long section crossing the line counts; v1 required a section's midpoint to be inside the 30% and could leave a few short bars as the only anchor. Every occurrence of a peak section's repeat group is also peak; repeated lower sections share one rank.
4. Peak sections keep the Planner density exactly. Their mean notes per beat is the peak reference.
5. Each other section is capped at `human_ratio(rank, target_star) × peak reference`. Its multiplier is `cap / section density`, clipped to [0.6, 1]. Notes and rows share the multiplier, so the local chord ratio is kept.

Nothing is ever raised. Quiet bars and zero budgets are untouched. The total budget is **not** preserved: lower sections lose notes by design. Because star rating follows the peak, the expected effect on star rating is small; see the real-song checks below for what was measured.

## Calibration

`keyloom_mania/resources/peak-allocation.json` (`peak-allocation-calibration-v2`) is produced by `scripts/calibrate_peak_allocation.py` from the evaluation reference profile (`keyloom evaluate reference ...`). For each human chart, 16-bar blocks with at least 12 active bars are sorted by notes per active bar and divided by the chart's own top-30% density. Median ratios at ranks 0.35–0.95 are stored per one-star bucket with at least 30 charts and interpolated by target star. The file records the profile's schema, feature schema and corpus digest. Rebuild it whenever those change; do not reuse it for another allocator version.

The current profile (4,225 usable train-split charts) gives middle-rank ratios near 0.87–0.94 at every star, and lowest-rank ratios that fall from 0.68 (≈2.4★) to 0.48 (≈8.4★): higher-star human charts have more section contrast.

The same file stores `star_from_peak_nps`, a least-squares fit of star rating on the log of each chart's top-30% block notes per second (the dominant tempo converts notes per bar). On this profile it is `star ≈ −4.91 + 3.42 · ln(peak nps)` with R² 0.91 and mean absolute error 0.34★. The same fit on whole-chart mean notes per second gives R² 0.85, which is consistent with star rating following the peak rather than the average.

## Generator star condition (`star_condition_id`)

The generator was trained with each chart's global star in every window. `global` (the command-line default; the workbench preselects `section`) keeps that. `section` (CLI `--star-condition section`) needs `peak-anchored` and gives each window the star a human chart would have if that section's shaped density were its peak:

`local star = target − min(1.5, −slope · ln(section density / peak reference))`, never below 2★.

Peak sections keep the target star; the star of a window is the mean over the candidate slots it newly decides (the continuation prefix is excluded). Quiet bars keep the target. `hierarchy.json` records `local_star` with the per-section stars and every window's condition (`section-local-star-v1`). This changes what the generator is asked for, not only the budget, so it can change onset and layout decisions; it is an experiment, not a verified local difficulty.

## Diagnostics

Every run, including `legacy` and `hierarchical`, writes `difficulty-placement.json` (`difficulty-placement-v2`, also inside the `.osz`). It reads the exported chart's 400 ms strain peaks from rosu-pp and maps them to bars. Tiers use the same audio sections and the same peak rule as the allocator, applied to the **unshaped** Planner density, so the three structure options of one song are tiered identically. It reports:

- `top_strain_peak_share`: the share of the 20 highest strain peaks that fall in peak sections, next to `peak_bar_share` (the share expected by chance). A low value means difficulty sits in verses.
- `lower_to_peak_strain_p90_median` and planned, shaped and exported `lower_to_peak_*_density_median`, to compare planned with realised contrast.
- per-section strain p90/max, planned/shaped/exported notes per beat and top-strain counts.

Bars planned as Stream are left out of every metric (`excluded_styles`, `excluded_bars`, `excluded_strain_sections`) because Stream decoding has a known generator bug. Section tiers and the allocator still use those bars; remove the exclusion once the fix has been re-measured. Strain carried over from an excluded bar into the next bar is not corrected.

Strain section alignment follows rosu-pp (sections end at multiples of 400 ms, starting from the second hit object). The report is observational, never feeds back into decoding, and a failure is recorded as `{"error": ...}` instead of failing the export. The workbench appends the top-strain share to the density diagnostic.

`hierarchy.json` contains `density_version` and, under `density`, the per-section tier (`peak`, `reduced`, `within_ratio`), rank, score components, human ratio, cap, multiplier and densities before and after, plus original and shaped budget sums. `bar-density.json` keeps the unshaped reference for every bar. The workbench subtitle reports the number of peak and reduced sections and the planned note change.

## Real-song check (2026-11-03)

Two songs, `music-base` r1, Jumpstream, target 6.0★, rule roles, fixed density, CUDA; one deterministic run per variant (the workbench HTTP path reproduced the CLI run exactly). Peak share is the share of the top 20 strain peaks inside peak sections; ratios are lower-section medians relative to peak sections.

| Song | Variant | Actual ★ | Exported notes | Top-20 peak share (peak bars) | Strain p90 ratio | Exported density ratio |
|---|---|---|---|---|---|---|
| 140 BPM | legacy | 3.44 | 2392 | 0.30 (0.32) | 0.99 | 0.94 |
| | hierarchical | 3.46 | 2372 | 0.30 | 0.98 | 0.91 |
| | peak-anchored | 3.26 | 2345 | 0.70 | 0.97 | 0.90 |
| | peak + section star | 3.32 | 2320 | 0.55 | 0.99 | 0.88 |
| 200 BPM | legacy | 4.46 | 4137 | 0.80 (0.54) | 0.77 | 0.65 |
| | hierarchical | 4.43 | 4116 | 1.00 | 0.76 | 0.63 |
| | peak-anchored | 4.42 | 3998 | 0.95 | 0.71 | 0.59 |
| | peak + section star | 4.54 | 3977 | 1.00 | 0.64 | 0.56 |

The 140 BPM rows exclude 2 Stream bars. The 200 BPM rows are `difficulty-placement-v1` and still include 10 Stream bars of 207 active, because their packages were no longer available to re-score.

- On the 200 BPM song, lower sections lost notes and strain while star rating stayed within 0.1★, as expected when star follows the peak.
- On the 140 BPM song the Planner already planned 0.80× contrast, so only two sections were lowered. Its difficulty was misplaced at chance level under legacy (0.30 vs 0.31).
- Both songs undershoot 6★ by 1.5–2.7★ in every variant. With Jumpstream the generator exports 30–60% more notes than the Planner budget, proportionally most in quiet sections (an intro planned at 2.0 notes/beat exported 3.3), which flattens contrast after planning; the peak-anchored option cannot remove it.
- A separate check on 27 validation songs (music-base r1, rule roles, fixed density, legacy structure, human-chart star targets, complete non-quiet bars) shows the overfill is style-specific but weak tracking is general. Across styles the per-song slope of exported on planned bar density has median 0.78 (1.0 = full tracking): the quietest quarter of bars exports 1.18× its plan and the densest quarter 0.94×. Jumpstream overshoots (total 1.54×, slope 1.21, low/high ratio 0.42 → 0.61); Minijacks (0.78×, slope 0.61) and Stream (0.88×, slope 0.53, low/high 0.30 → 0.51) undershoot and flatten most; Gluts, Handstream, Quadstream and Chordstream stay near 0.97–0.98× in total. The median total over all 27 songs is 0.98×. These pooled figures predate the Stream exclusion and still include the Stream songs; the Stream figures themselves are on hold until its generator bug is fixed.
- No reductions or drops were reported, and every export passed the Minijacks audit.
- The project owner listened to and played the new Jumpstream charts and reported that thinner verses still follow the song and choruses clearly burst.

## Minijacks and Chordjacks check (2026-11-04)

Two validation songs per style, chosen in advance by rule (closest to 140 BPM, then highest style share, different song groups). `music-base` r1, no secondary style, rule roles, fixed density; target star, BPM and offset from the human chart. Sections and tiers are shared by every row, including the human chart. Cut is reductions and drops across the prepare, layout and export stages. Every export passed the Minijacks audit. No chart in this section has been listened to.

| Song | Variant | Actual ★ | Planned notes | Exported notes | Cut | Top-20 peak share (peak bars) | Strain p90 ratio | Exported density ratio |
|---|---|---|---|---|---|---|---|---|
| Minijacks A, 140.0 BPM | legacy | 4.22 | 3608 | 3136 | 31 | 0.90 (0.34) | 0.58 | 0.39 |
| | peak-anchored | 4.22 | 3608 | 3136 | 31 | 0.90 | 0.58 | 0.39 |
| | peak + section star | 4.21 | 3608 | 3029 | 23 | 1.00 | 0.56 | 0.38 |
| | human | 4.79 | — | 1800 | — | 1.00 | 0.40 | 0.00 |
| Minijacks B, 140.6 BPM | legacy | 4.45 | 3648 | 3233 | 26 | 0.05 (0.31) | 0.99 | 0.80 |
| | peak-anchored | 4.46 | 3559 | 3223 | 23 | 0.00 | 0.99 | 0.80 |
| | peak + section star | 4.61 | 3559 | 3189 | 68 | 0.95 | 0.92 | 0.75 |
| | human | 7.17 | — | 3167 | — | 0.95 | 0.50 | 0.35 |
| Chordjacks A, 140.0 BPM | legacy | 6.16 | 3611 | 4199 | 10 | 0.65 (0.83) | 0.91 | 0.86 |
| | peak-anchored | 6.17 | 3459 | 4203 | 9 | 1.00 | 0.97 | 0.78 |
| | peak + section star | 6.16 | 3459 | 4172 | 5 | 1.00 | 0.92 | 0.73 |
| | human | 7.37 | — | 4633 | — | 1.00 | 0.76 | 0.61 |
| Chordjacks B, 140.0 BPM | legacy | 5.38 | 4310 | 4462 | 17 | 1.00 (0.71) | 0.92 | 0.82 |
| | peak-anchored | 6.06 | 4135 | 4577 | 15 | 1.00 | 0.71 | 0.62 |
| | peak + section star | 5.62 | 4135 | 4383 | 21 | 1.00 | 0.76 | 0.62 |
| | human | 6.10 | — | 4210 | — | 1.00 | 0.46 | 0.43 |

- Chordjacks gained clear contrast on both songs, moving toward the human ratio without reaching it. Peak sections cover 71–83% of these songs, so the top-20 share says little there; read the density and strain ratios.
- Minijacks barely responded to allocation alone. On A the Planner was already below the human ratio, so nothing was capped and the chart is identical to legacy. On B the section star made the difference (top-20 share 0.00 → 0.95), at the cost of more cut notes (23 → 68).
- Minijacks A's human chart leaves the lower sections almost empty and has 1800 notes, so its ratios are not a useful target. Minijacks B's human chart is 45% Minijacks and 40% Chordjacks and reaches 12.7 notes per beat in its peak; the single-style Minijacks run stays near 8.0, so its 7.17★ is not a fair target for that setting.
- Star rating can swing without any budget change. On Chordjacks B, bars 73–102 kept their budget, style and 3.88 rows per beat, but chords grew from 9.3 to 11.6 notes per beat, which accounts for most of the 5.38 → 6.06★ rise. A likely cause is the count head's dependence on earlier counts; this was not verified. With one run per variant, star differences of this size should not be read as an effect of the option itself.

## Limits

- The 16-bar block grid is not phrase-aligned; the ratios are a shape prior, not a quality label.
- Section intensity uses broadband energy and the Planner's density. It is not verified verse/chorus labelling.
- With the default `global` star condition the generator receives the song-wide star in every window, so pattern complexity in lowered sections is unchanged.
- Promotion requires listening, phrase review and playtesting, not lower pressure or smaller star error. Listening so far covers Jumpstream only.

## Planner training

The root cause in the base Planner: one star value is broadcast to every bar, and the per-bar log-density regression plus the adjacency and repetition smoothness terms pulls verses toward the chart median. `planner_loss` accepts two opt-in weights; absent or zero, the objective is bit-for-bit the original:

- `density_quantile`: smooth-L1 (`beta` 0.05, near-L1) between each chart's predicted and true 10th/50th/90th percentile of log notes per beat (charts with at least 8 valid bars).
- `section_contrast`: for charts with `section_ids` (per bar, -1 where none), a gap loss on all pairs of section-mean log densities plus a 0.15 margin rank loss on pairs whose true gap exceeds the margin. The ids must come from audio structure (for example `section_phrase_structure` on the bar features, with empty target bars as quiet). The dataset's `is_section_start` marks timing sections only, so a constant-tempo song has a single start; the loss ignores it.

### Candidate study (2026-11-03, validation split only)

`arrangement-r1` is bit-identical to the archived structured-v2 run (seed 42, step 3000: AdamW 5e-4, batch 8, 384-bar crops, `pattern_positive` 0.5, `pattern_mean` 1.0, `structure_adjacent` / `structure_repeat` / `quiet_energy` 0.15). Each arm reused that frozen dataset and training loop unchanged, swapping in this repository's `planner_loss` and audio `section_ids`, with seed 42 and 3,000 steps. No sealed-test chart was read. Metrics are medians over 513 validation charts: p90 bias is predicted minus true 90th-percentile notes per beat; flattening is the predicted minus true ratio of lower-section to peak-section density on the true section ranking.

| Arm | Extra weights | MAE | Corr | p90 bias ≥5.5★ | p90 bias <4★ | Flattening ≥5.5★ |
|---|---|---|---|---|---|---|
| r1 (released) | — | 1.385 | 0.768 | −1.73 | +0.09 | 0.066 |
| baseline (retrained r1 contract) | — | 1.411 | 0.770 | −1.83 | +0.22 | 0.074 |
| contrast | quantile 0.5, contrast 0.5 (`beta` 1) | 1.406 | 0.784 | −2.40 | −0.14 | 0.061 |
| contrast-lowsmooth | as above, structure 0.05 | 1.488 | 0.779 | +0.04 | +0.56 | 0.063 |
| contrast-l1 | quantile 1, contrast 1 (`beta` 0.05) | 1.419 | 0.763 | −1.73 | −0.36 | 0.092 |
| contrast-l1-lowsmooth | as above, structure 0.05 | 1.399 | 0.776 | −0.32 | +0.35 | 0.097 |
| **q-lowsmooth** | quantile 1 (`beta` 0.05), structure 0.05 | **1.356** | **0.783** | **−0.71** | **+0.04** | 0.069 |
| q-rank-lowsmooth | + rank hinge only, 1 | 1.395 | 0.791 | −0.37 | +0.73 | 0.062 |

- r1 under-plans the peak of high-star charts by 1.7 notes per beat and plans their lower sections flatter than humans (0.80 vs 0.70 of the peak). Low-star charts are nearly unbiased.
- `q-lowsmooth` is the most balanced: lowest MAE, and it halves the high-star peak deficit without overshooting low stars. The run-to-run noise between r1 and its retrained baseline is about 0.1 notes per beat.
- No arm reduced high-star flattening beyond noise. Under near-L1, the all-pairs gap term made it worse, because most section pairs have small true gaps and an L1 fit pulls every gap toward them. Do not use `section_contrast` with near-L1 gaps; section contrast still comes from the peak-anchored allocator.
- On the two real songs above, `q-lowsmooth` planned 15–26% fewer notes than r1, but the export changed by only 3–5% and star rating by under 0.1★ (3.38 vs 3.44, 4.42 vs 4.46). The generator's weak per-bar tracking of the density condition, not the Planner budget, sets most of the final density and difficulty.

No candidate is promoted or registered in the catalog. A planner revision is worth packaging only after the generator tracks its density condition (slope near 1 for every style); until then its effect on charts is small. Promotion would need a new catalog identity, `arrangement-r1` kept as baseline, density adapters retrained or marked unavailable for the new digest, and listening and playtesting.

A further step, not implemented, is to separate shape from level: predict a star-independent density curve relative to the chart's peak, and predict the peak level from star and tempo with a quantile (not mean) regression, then combine them with the peak-anchored allocator. That changes the model architecture and needs its own version and catalog entry.
