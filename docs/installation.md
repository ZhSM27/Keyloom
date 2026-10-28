# Installation and operation

Python 3.11–3.13 is supported by the package metadata; Python 3.12 is the primary validation target. Create a virtual environment and install `python -m pip install -e ".[beats]"`. Match torch and torchaudio versions. For NVIDIA inference, install the appropriate CUDA build using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing this package.

## Resources

1. Download [keyloom-models-r1.zip](https://github.com/ZhSM27/Keyloom/releases/download/v0.1.0/keyloom-models-r1.zip) from the [v0.1.0 Release](https://github.com/ZhSM27/Keyloom/releases/tag/v0.1.0). Run `keyloom assets install <bundle.zip>`.
2. Run `keyloom assets fetch-audio --beats`. Downloads use pinned SHA-256 values. MusicFM is required; Beat This! is optional for automatic BPM/offset detection. Without it, enter timing manually.
3. Run `keyloom doctor --load-models --device cpu`. This checks the complete model resource catalog and MusicFM files, then loads all four generators. It does not measure musical quality or execute the complete audio encoder. Use `--device cuda` to exercise native model loading on the GPU.
4. Run `keyloom studio` and open `http://127.0.0.1:8788/`.

The source tree and wheel do not contain weights. An absent or corrupt resource produces an installation error; a different model is never silently substituted. Close the workbench before installing or upgrading a resource bundle. Verified resources are replaced individually; interrupted installation can be repaired by rerunning the installer.

## Storage

`KEYLOOM_HOME` sets the application directory. The default is `%LOCALAPPDATA%/keyloom` on Windows and `${XDG_DATA_HOME:-~/.local/share}/keyloom` on other systems. Set the variable before starting Python.

```text
<application directory>/
  assets/generator/       # four generators and shared inference statistics
  assets/planner/         # base planner, adapters and normalization
  assets/audio/musicfm/   # pinned upstream audio encoder resources
  assets/audio/beat-this/ # optional beat model and phase calibration
  generations/<id>/      # uploaded audio, chart, diagnostics and package
```

Downloads require internet access. Once installed, generation runs locally. Original input files are copied to the generation directory. The application does not automatically delete uploads; remove unwanted generation directories yourself after jobs finish.

## CPU and timing

The HTTP workbench uses CUDA. CPU users can run `keyloom generate song.wav --bpm 120 --offset 0 --device cpu`. CPU inference is substantially slower. Use `--structure hierarchical` to evaluate the latest planner and `--model count-rollout` for that generator candidate.

Offset is the timestamp, in milliseconds, of the selected first beat; negative offsets are accepted within the documented input range. BPM is constant throughout the output. Beat detection reports fit quality, not a calibrated probability of correctness. Variable tempo and ambiguous beat grouping require manual review.

An occupied port is reused only when its health endpoint reports this package's runtime version. A different service results in an error; the launcher never kills it or opens another workbench port.
