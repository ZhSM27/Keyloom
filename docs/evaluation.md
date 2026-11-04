# Chart evaluation

`keyloom evaluate` screens exported charts against local human charts. It answers "does this chart look like what human mappers write at a similar difficulty, tempo and style mix?" It does not replace listening, phrase review or playtesting, and no model may be promoted on its scores alone.

```sh
keyloom evaluate reference "<osu! Songs folder>" "<osu!lazer files folder>"
keyloom evaluate chart baseline.osz candidate.osz --markdown report.md --json report.json
```

The reference command scans folders for 4K mania charts, including the extensionless osu!lazer file store, and reads corpus manifests given with `--manifest` (chart and audio paths relative to the manifest). Manifests contribute only the `train` split unless `--split` names others; never add held-out test splits. It excludes charts written by this project (creator, version or tag markers), duplicates, long-note-heavy charts (>10% holds by default), charts shorter than 200 notes and charts with many timing sections. The profile is written to `<application directory>/evaluation/human-reference.json.gz`. It stores aggregate observations and SHA-256 identities only, never chart text or paths. Build it from your own library; it is calibration data and stays outside the source tree.

`evaluate chart` accepts `.osz` packages, `.osu` files or a generation folder. Packages produced by the workbench also contribute their bar plan, execution diagnostics and target star. Exit status 2 means a hard constraint failed.

## What is measured

| Axis | Observations |
| --- | --- |
| Hard constraints | same-column hits under 40 ms, 2–10 ms split rows, parse issues, and the export Minijacks audit when Minijacks are planned |
| Style realization | the planned label's pattern rule must hold in each complete exported bar; each style is scored relative to the human ceiling (phrase-majority self-consistency of human charts) |
| Within-style naturalness | each style's bars compared with human bars of the same style: 2/4-beat hand rows, notes and column hits, same-hand gaps, chord sizes, transitions, row gaps and snaps |
| Hand pressure | per-hand rows and notes in the last 2 and 4 beats, per-column hits, notes per second, local hand bias, same-column/same-hand gaps, participation runs and the busiest column's share of stream bars (anchors) |
| Layout | chord sizes, adjacent-row transitions (sizes, overlap, hand relation), jack transitions, column use, ordinary long jacks, near-copied adjacent bars, bar motif variety, and the lengths of same-column runs, trills and rolls |
| Rhythm | snap divisions, row gaps, bar density variation, constant-interval stream share, chords on 1/1 or 1/2 positions within streams, and isolated grace gaps |
| Music | onset precision, timing and beat-selection lifts against shifted copies of the chart, strong-onset recall, energy–density correlation, and accents: chords on strong onsets relative to singles (accent lift), strong onsets covered by chords, chord size against onset strength, chords and off-stream rows without a supporting onset, and beats holding more rows than onsets plus one (needs audio) |
| Execution | planned versus exported notes and rows, reductions and dropped rows, copied from the package diagnostics |

Hand-pressure observations use only the recent 2 and 4 beats; a hit four or more beats old is the same as no hit. No metric prefers lower pressure or balanced hands. Global hand share is shown for reference only.

## Pattern rules

Bar styles come from `community-pattern-rules-v2-tech` (`keyloom_mania/evaluation/definitions.py`), written from the osu! wiki stream, trill and chord pages, the osu!mania ranking-criteria glossary and Etterna terminology. They replace the YAVSRG motif detector inside the evaluator. That detector ignores row timing, never recognises single-column jacks, and misses one-jump-per-beat jumpstreams, pure rolls and pure trills; on 150 local human charts it left 48% of active bars without any label covering half the bar, against 14% for these rules (mostly irregular rhythms that are not streams by definition). The vendored detector itself is unchanged and still serves the research scripts.

Rows closer than half a beat are linked. A stream row lies in a constant-interval run lasting at least one beat; a stream bar has at least 60% stream rows. A bar needs four rows to carry a style.

| Label | Rule |
| --- | --- |
| Stream | stream bar, at most one minijack, almost no chords |
| Chordstream | stream bar with two or more chords and jacks only beside quads |
| Jumpstream | chordstream whose chords are jumps (two or more per bar) |
| Handstream | chordstream with two or more hands per bar, jumps allowed |
| Quadstream | stream bar with two or more quads; quads may jack their neighbours |
| Rolls | single-note sweeps over adjacent columns cover half of a stream bar |
| Trills | single notes alternating between two columns cover half the bar in runs of four or more |
| Minitrills | three or more short trill fragments in a stream bar (no community definition exists) |
| Jumptrill | jumps alternating between hands (12/34) cover half the bar |
| Split Trill | jumps that each need both hands (13/24, 14/23) alternate over half the bar |
| Jacks | half the rows repeat a column from the previous row and a run reaches three |
| Minijacks | two or more same-column runs of two or three notes, none longer |
| Longjacks | a same-column run of five or more notes covers a quarter of the bar |
| Chordjacks | mostly chords, half of the rows chord-jacking the previous chord |
| Gluts | mostly jumps whose minijacks shift columns, no run beyond three |
| Tech | a dense bar (at least six rows, three quarters linked) that mixes 1/4-family and 1/3-family positions (polys), changes interval by awkward non-integer ratios in a quarter of its gap pairs, or bursts three rows at half its usual gap |

Tech has no agreed definition either. The rule follows Etterna's technical skillset (awkward patterning from any category, naming bursts, minijacks, gluts and polys), MinaCalc's chaos measure of consecutive-gap ratios and community descriptions of rhythm maps that chain 1/3, 1/4 and 1/6 deviations. It forms its own style family and takes precedence over the other labels. Tech is evaluator-only: the released planner and generators keep the fifteen-label vocabulary, so a plan never requests it. On the same 150 human charts 13% of active bars are Tech and bars without any label fall to 10%.

Several labels may hold at once. The dominant label (most specific first: Tech, Jumptrill, Split Trill, Gluts, Chordjacks, Longjacks, Quadstream, Handstream, Jumpstream, Rolls, Trills, Jacks, Minijacks, Minitrills, Chordstream, Stream) groups bars for within-style comparison and style-family neighbour matching. Plan realization accepts a bar when its planned label holds, whatever else holds.

## Calibration

For each evaluated chart the 40 nearest reference charts are chosen by calculated star rating, tempo and detected style-family mix. Their 16-bar blocks are cut into segments as long as the evaluated chart, so a 32-bar excerpt is compared with 32-bar human segments. Each metric is placed in that human sample as a percentile. Two-sided metrics are unusual in either direction; distribution divergences (Jensen–Shannon against the pooled neighbours, leave-one-chart-out for human segments) are unusual only when high.

An axis score is the percentage of human samples that are at least as atypical as the chart. Each metric contributes `-log` of its tail probability. Human samples are scored the same way, leave-one-out, so human charts spread over 0–100 rather than clustering at 100. Whole-chart naturalness uses the hand-pressure, layout and rhythm metrics on matched-length segments. Within-style naturalness compares windows of up to 32 bars of one style with windows of the same style from the neighbours, and weights styles by their bar count. The headline **naturalness** is the mean of the two. **Style realization** is reported beside it and never averaged in. The music axis is calibrated against the nearest reference charts that have audio (same star, tempo and style-mix distance) once at least eight exist; otherwise its values are printed as uncalibrated. When calibrated it joins whole-chart and within-style naturalness in the headline mean. Strong onsets are the top quarter of onset strengths within the chart span; support means an onset within 30 ms.

## Limitations

- Pattern terms are not standardised across communities. The rule thresholds follow the cited definitions but were checked only on local charts; human ceilings use each chart's own phrase-majority labels and are optimistic.
- The planner and generator style vocabulary was learned with the old detector's labels, so a planned "Jumpstream" may have been trained toward dense jumpstream; the new rules judge it against the community definition.
- The reference reflects whatever charts you have installed, of unknown curation. Neighbour sets are small for rare tempos or difficulties.
- osu!lazer stores audio without names, so lazer charts give no music-axis calibration; stable song folders and corpus manifests do.
- Accent measures use one spectral-flux onset detector; fast streams can merge onsets, and the detector does not separate drums from melody.
- Rebuild the reference whenever `FEATURE_SCHEMA` changes. Profiles built with other feature semantics are refused rather than reinterpreted.
