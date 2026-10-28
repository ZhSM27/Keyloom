# Releasing

1. Run the portable test suite and `python scripts/check_release.py`.
2. Build with `python -m build`; inspect both distributions with `python scripts/check_release.py dist`.
3. Run `python scripts/package_source.py` to create a clean GitHub source ZIP. It uses an explicit source allowlist rather than copying the working directory.
4. Obtain the maintainer's portable inference assets, verify them against `keyloom_mania/resources/catalog.json`, and create a ZIP whose root contains exactly the catalogued files. Do not include upstream MusicFM weights, music, training data or private experiment metadata. The supplied release bundle is named `keyloom-models-r1.zip`.
5. Install the wheel in a clean environment, install the bundle, fetch upstream audio assets and run the doctor. Run a complete generation smoke test outside the source tree. Report its limitations separately from musical-quality evaluation.
6. Publish the source repository, tag and built distributions, and attach the matching model ZIP to the same GitHub Release. Enable private vulnerability reporting if desired. Repository hosting and uploads are a maintainer action, not part of the local build.

Never upload the entire local project directory as an unchecked archive. Only public sources, licenses, documentation and tests belong in Git. Assets use a separate release bundle and app data directory. Record public hashes and validation results; retain private migration notes outside the repository.
