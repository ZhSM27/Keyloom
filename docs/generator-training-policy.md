# Generator and evaluation policy

Musical naturalness is the primary objective: alignment with music, real-chart continuation and key-placement distributions, phrase expression and playtesting. Lower pressure, balanced hands, style-hit rate or a single training loss cannot substitute for those observations.

Hand pressure uses only the recent **2 and 4 beats**, including features, scoring/losses, calibration and acceptance. Do not accumulate 8/32-beat compensation debt across phrases. Pressure is auxiliary: local hand preference, long jacks, AAB and syncopation can be musically appropriate. A pressure improvement alone must not replace a more natural baseline.

Musical organization can use long context. Do not constrain the entire model to four beats. Counts should follow music; always compare planned counts with actual export reductions and dropped rows.

Preserve gameplay constraints: Minijacks disallow four consecutive sixteenth-note hits on a column and allow at most five consecutive participating rows per hand. Ordinary jack runs should rarely be very long; Longjacks has no hard length cap.

Historical 20-dimensional pressure inputs used 2/8/32-beat semantics. Changing only the windows is not a valid migration of adapters, thresholds, prefixes or scores. New training must version the input semantics, recalibrate and resample prefixes consistent with the active decoder. Frozen historical experiments may be kept externally for comparison and must be identified by their original semantics.

Research candidates may be prepared and evaluated without an additional approval ritual. Report limitations and do not promote them without musical evidence. Keep comparison checkpoints immutable and inspect concurrent planner changes before editing.
