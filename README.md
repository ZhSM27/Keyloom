# Keyloom · 织键

把音乐织成谱面。本地运行的 osu!mania 4K 谱面生成工作台。将音乐特征、段落/乐句规划、主副键型编排与押数/排键生成接入同一条流程，输出可导入的 `.osz` 和生成诊断。

安装包名为 `keyloom-mania`，命令为 `keyloom`，Python 模块为 `keyloom_mania`。

[English](README.en.md) · [安装说明](docs/installation.md) · [模型说明](docs/models.md) · [架构](docs/architecture.md)

这是研究软件。自动生成的自然度、音乐对应和可玩性仍需试听、试玩；目标星级是模型条件，导出星级另行计算，二者不保证相等。当前支持恒定 BPM 的 4K 点击谱面，不支持完整变速曲线和长条生成。

## 快速开始

需要 Python 3.11–3.13；主要验证环境为 Python 3.12。先创建并激活虚拟环境，然后安装：

```sh
git clone https://github.com/ZhSM27/Keyloom.git
cd Keyloom
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[beats]"
```

GPU 工作台需要匹配的 CUDA 版 PyTorch 和 torchaudio，安装方法见 [PyTorch](https://pytorch.org/get-started/locally/)。CPU 用户可用离线生成命令。

从 [v0.1.0 Release](https://github.com/ZhSM27/Keyloom/releases/tag/v0.1.0) 下载匹配的 [keyloom-models-r1.zip](https://github.com/ZhSM27/Keyloom/releases/download/v0.1.0/keyloom-models-r1.zip)，然后：

```sh
keyloom assets install keyloom-models-r1.zip
keyloom assets fetch-audio --beats
keyloom doctor --load-models
keyloom studio
```

源码包不含权重或音乐。模型包约 137 MiB，解压约 151 MiB；MusicFM 另需约 1.3 GB 下载。`--beats` 下载可选的自动拍点模型；不安装拍点模型也可以手动填写 BPM/offset。

工作台唯一地址：**http://127.0.0.1:8788/**。Windows 也可运行 `studio.bat`；重复启动复用同版本工作台，不新开备用端口。

CPU 离线示例：

```sh
keyloom generate song.wav --bpm 120 --offset 0 --star 5 --style Jumpstream --device cpu
```

## 保留的模型

| 公共 ID | 显示名 | 用途 |
| --- | --- | --- |
| `music-base` | 音乐押数 · 基线 r1 | 默认发布基线 |
| `count-lookahead` | 计数前文 · 前瞻对照 r1 | 前瞻开关对照 |
| `count-teacher` | 计数前文 · 真值前文 r1 | 真实前文训练研究对照 |
| `count-rollout` | 计数前文 · 生成前文 r1 | 生成前文训练研究候选 |

每个系列独立使用 `r1`、`r2` 修订号。所有选项均为研究版本，默认基线不代表质量认证。Planner 保留最新 `hierarchical` 段落—乐句规划与 `legacy` 原方案；条件编排保留标准/主副均衡，密度保留原密度/段落对比。最新方案尚未自动晋升默认。

## 目录与数据

```text
keyloom_mania/
  studio/                  # 配置、注册、加载、契约、生成、导出
  resources/catalog.json  # 模型身份、相对路径、大小、SHA-256
  generation_studio_web/   # 单页工作台
  _vendor/                # 第三方代码及许可证
  *.py                    # Planner、生成模型与推理基础组件
tests/                    # 不依赖私有数据或权重的回归测试
docs/                     # 安装、模型、架构和研究约束
scripts/                  # 发布校验及源码打包
.github/workflows/        # Windows/Linux CI
```

资产和输出默认放在系统用户数据目录，详见[安装说明](docs/installation.md)。可用 `KEYLOOM_HOME` 指向自己的目录。仓库不包含训练数据、原始音频、机器环境、实验输出或个人路径。

## 开发

```sh
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python scripts/check_release.py
python -m build
```

源码采用 [MIT](LICENSE)；第三方组件和外部权重见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。发布步骤见 [docs/releasing.md](docs/releasing.md)。
