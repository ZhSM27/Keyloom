# Engineering and model changes

- Musical naturalness takes priority over pressure scores, hand balance, style-hit metrics and training loss. Verify music alignment, phrase continuation and actual playability.
- Hand-pressure observations use only the recent 2 and 4 beats. Longer musical context is valid; do not turn the pressure horizon into a model context limit.
- Do not promote a candidate solely for lower pressure. Retain local hand bias and intentional jack patterns; do not impose a fixed left/right ratio.
- Preserve Minijacks limits: no four consecutive sixteenth-note hits on one column, at most five consecutive participation rows per hand. Ordinary long jack runs should remain rare; Longjacks has no hard run-length cap.
- Keep planned versus exported counts, reductions and dropped rows visible. Preserve the baseline when quality is uncertain.
- Version feature semantics, calibration data and decoder-matched prefixes together. Do not reuse historical 2/8/32-beat thresholds as 2/4-beat calibration.
- Inspect current changes before editing. Never overwrite parallel planner work. Keep historical checkpoints and frozen comparisons outside the public source tree.
- Keep one workbench port (8788), model identities in the catalog, and machine paths out of source. Run portable tests and release checks before delivery.
