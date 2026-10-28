# Release validation

Validation for the 0.1.0 source and r1 asset bundle:

- Portable unit and regression suite: 137 tests, with no private training corpus or model downloads.
- Every released generator tensor compared equal to its retained source checkpoint; the inference-only rewrite removes training state and machine metadata.
- All catalogued assets pass SHA-256 checks, and all four generators load natively with strict state dictionaries.
- The wheel is installed into a separate directory and run from outside the checkout using existing third-party dependencies. This checks package independence from project source and private research modules; it is not a fresh operating-system installation.
- Source, wheel and source distribution are scanned for absolute machine paths, credentials, weights, audio and unexpected large files. The GitHub Actions workflow repeats portable tests and package builds on Windows/Linux; hosted CI has not yet run before publication.

Full audio-to-export smoke cases use 16 seconds of synthesized audio at explicit 120 BPM:

| Generator | Device | Structure | Role / density | Style | Exported notes / rows |
| --- | --- | --- | --- | --- | --- |
| music-base | CPU | legacy | rule / fixed | Jumpstream | 158 / 128 |
| music-base | CUDA | hierarchical | standard / contrast | Jumpstream | 160 / 128 |
| count-lookahead | CUDA | legacy | rule / fixed | Minijacks | 26 / 20 |
| count-teacher | CUDA | hierarchical | balanced / contrast | Jumpstream | 160 / 128 |
| count-rollout | CUDA | hierarchical | rule / fixed | Longjacks | 34 / 31 |
| music-base, count-reference budget | CUDA | legacy | rule / fixed | Jumpstream | 160 / 128 |

Each case produced an `.osz` accepted by the export validator and difficulty calculation. The Minijacks smoke case passed column/hand run audits. A paired run of the original and portable baseline with identical hierarchical planner settings produced identical exported hit objects and timing points.

The live workbench also passed upload, automatic timing detection, generation and download over HTTP, with cross-origin writes rejected. Exclusive address binding prevents multiple Windows listeners from sharing the workbench port; repeated starts reuse a matching running release.

These are engineering smoke checks, not listening or playability evaluation. Synthetic audio does not establish performance on real songs, target-star accuracy, generalization or long-song memory behavior. CPU and CUDA can produce different discrete decoding decisions; bitwise equality across devices is not promised. Full training reproduction is outside this inference release.
