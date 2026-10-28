# Third-party notices

Project source is MIT-licensed. This does not replace upstream component terms or confer rights to users' input music and charts.

## Pattern analysis

`keyloom_mania/_vendor/omtk_pattern/` contains the pattern-analysis core from [LeoBlackMT/nonebot-plugin-osumania-toolkit](https://github.com/LeoBlackMT/nonebot-plugin-osumania-toolkit), commit `fff821e54e34c7a9649588ec758ead5353e9f2f5`. The recognition module credits [YAVSRG/Interlude Prelude](https://github.com/YAVSRG/YAVSRG), commit `9786913f5e52e6ef0fb51fdacc8c7c61f42728d0`. Both relevant MIT license texts are preserved beside the vendored code. Imports and configuration wiring are adapted; provenance and original source hashes are retained in its `provenance.json`.

## MusicFM

The frozen audio frontend uses [MusicFM](https://github.com/minzwon/musicfm) source revision `b83ebedb401bcef639b26b05c0c8bee1dc2dfe71` and [published MusicFM-FMA weights](https://huggingface.co/minzwon/MusicFM) revision `4513b38bc25ad1d227b1980819b9691ba97f4d87`. The standard modules and published weight metadata use MIT terms. The upstream [license](https://raw.githubusercontent.com/minzwon/musicfm/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/LICENSE) distinguishes the Apache-2.0 Flash Conformer module, which is not used or downloaded here.

The installer fetches verified upstream source with its LICENSE and README. `music_features.py` is an inference wrapper using those frontend/convolution modules and the installed Transformers Conformer implementation. Configuration comes from the pinned Facebook wav2vec2 Conformer configuration listed in `audio_assets.py`; its pretrained weights are not downloaded. All source and weight digests are pinned. MusicFM weights are not included in the project source or generator bundle.

## Beat This!

Optional automatic timing uses [Beat This!](https://github.com/CPJKU/beat_this), package `1.1.0`, by Francesco Foscarin, Jan Schlüter and Gerhard Widmer. Its [MIT license](https://github.com/CPJKU/beat_this/blob/main/LICENSE) is retained by the installed package. The optional `final0` checkpoint is downloaded from the upstream JKU host and verified with SHA-256 `8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`. Inference uses `Audio2Beats(dbn=False)`; madmom models are not used.

## Installed dependencies

PyTorch, torchaudio, NumPy, PyAV/FFmpeg, Transformers, einops, mutagen and rosu-pp-py are installed as dependencies and retain their own licenses. These dependencies are not relicensed by this repository. Consult their installed distribution metadata when redistributing an environment.
