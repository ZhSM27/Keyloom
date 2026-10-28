# Keyloom

Python distribution: `keyloom-mania`. Command: `keyloom`. Python module: `keyloom_mania`.

A local osu!mania 4K generation workbench combining audio features, section/phrase planning, style arrangement, count prediction and key placement. Exports an `.osz` plus execution diagnostics. This is a research release: musical quality and playability require listening and playtesting.

See the [Chinese README](README.md) for model options and [installation](docs/installation.md) for details.

```sh
git clone https://github.com/ZhSM27/Keyloom.git
cd Keyloom
python -m venv .venv
# Activate the environment, then:
python -m pip install -e ".[beats]"
keyloom assets install keyloom-models-r1.zip
keyloom assets fetch-audio --beats
keyloom doctor --load-models
keyloom studio
```

Download the matching [model bundle](https://github.com/ZhSM27/Keyloom/releases/download/v0.1.0/keyloom-models-r1.zip) from the [v0.1.0 Release](https://github.com/ZhSM27/Keyloom/releases/tag/v0.1.0). MusicFM is fetched separately with pinned SHA-256 verification. A CUDA-enabled PyTorch installation is required for the workbench; offline CPU generation is supported:

```sh
keyloom generate song.wav --bpm 120 --star 5 --device cpu
```

The only workbench endpoint is `http://127.0.0.1:8788/`. Resources and outputs live outside the checkout in the platform user-data directory, overridable with `KEYLOOM_HOME`.

Public generators: `music-base`, `count-lookahead`, `count-teacher`, `count-rollout`, all revision `r1`. The hierarchical planner remains opt-in alongside the baseline. Current output is constant-tempo, four-key tap charts; long notes and variable-tempo maps are unsupported. Requested difficulty is a condition, not a guaranteed output rating.

The repository includes inference and portable regression tests, not the private training corpus or historical experiment scripts. MIT-licensed source; see [third-party notices](THIRD_PARTY_NOTICES.md).
