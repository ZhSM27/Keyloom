# Model cards and naming

Public identities use a descriptive family ID and a family-scoped revision. `music-base` at revision `1` exports as `music-base-r1`. Checkpoint SHA-256 and model identity accompany exported charts. Internal tensor-feature schema versions are retained for compatibility and are not public experiment version numbers.

| Component | ID | Revision | Role |
| --- | --- | --- | --- |
| Generator | music-base | r1 | retained default, learned music-conditioned counts |
| Generator | count-lookahead | r1 | count-prefix architecture with lookahead policy |
| Generator | count-teacher | r1 | true-prefix training comparison |
| Generator | count-rollout | r1 | generated-prefix training candidate |
| Role planner | rule | r1 | rule-based baseline |
| Role planner | standard / balanced | r1 | latest conditional role models |
| Density planner | fixed / contrast | r1 | base density / section contrast adapter |
| Structure | legacy / hierarchical | r1 | baseline / latest section–phrase hierarchy |

The catalog is the authority for file names, revisions, architecture IDs, sizes and digests. A density adapter must also match the base planner digest. Released assets contain inference configuration, normalization and weights, without optimizer state, RNG snapshots, chart lists or machine paths. The source distribution does not include the training corpus or full training pipeline; it supports inference reproduction, not independent reproduction of training.

## Research status

All four generators remain research models. Teacher-prefix and rollout-prefix variants have not demonstrated sufficient musical improvement to replace the retained baseline. Better count loss alone does not establish fewer reductions or more natural patterns. The hierarchical planner is opt-in: it adds section/phrase structure, but its impact depends on the selected generator, arrangement and density controls.

`count-reference` density execution currently supports only `music-base` with Jumpstream. It is not a general solution to planned-versus-exported count mismatch. The unavailable feasibility adapter is not presented as a trained model. Execution reports preserve planned counts, decoded counts and export reductions/drops separately.

## Inputs, limitations and evaluation

Input: audio, constant BPM/offset, target star condition (2–9), a primary style, up to three compatible secondary styles. Output: four-key tap charts. Audio is represented with acoustic, local-rhythm and frozen MusicFM features. Base arrangement, conditional role and density models feed the generator through validated contracts.

Finite hand observations cover 2 and 4 beats. Musical phrase and repetition context may be longer. Minijacks constraints and Longjacks freedom are retained. Do not infer musical quality from a lower pressure statistic or balanced hands.

The project does not provide a population-wide quality score, generalization guarantee or data license for third-party music/charts. Evaluate with music alignment, natural continuation, phrase expression, playtests, and actual export counts. See the training policy before proposing further training. Original pretrained audio components retain their upstream terms.
