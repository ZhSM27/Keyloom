# Contributing

Create a virtual environment, install `python -m pip install -e ".[dev]"`, and run `python -m unittest discover -s tests -v`. Tests require no model downloads or private corpus. Use `python scripts/check_release.py` before submitting a change.

Describe the behavior changed, relevant validation and limitations. For inference changes, report music alignment, playability, planned/exported counts and reductions, and compare against the retained baseline. A passing unit test does not establish musical quality.

Model revisions require a new catalog entry or revision, verified asset hashes and an updated model card. Do not commit weights, copyrighted songs, training corpora, generated charts, personal paths or environment credentials. Training-related changes follow [the training policy](docs/generator-training-policy.md).
