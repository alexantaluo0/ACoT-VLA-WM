# ACOT-VLA-WM: Precise Robotic Subgoal Generation and Execution

[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://www.acotvla-wm.xyz)
[![arXiv](https://img.shields.io/badge/arXiv-2601.11404-b31b1b.svg)](https://arxiv.org/pdf/2601.11404v2)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**ACOT-VLA-WM** is an improved variant of [**ACoT-VLA**](https://arxiv.org/abs/2601.11404v2) that integrates a **Predictive World Model** into the Action Chain-of-Thought framework, enabling precise robotic subgoal generation and execution on long-horizon, high-precision manipulation tasks.

> 📖 Full technical details, demos, and evaluation videos: **[acotvla-wm.xyz](https://www.acotvla-wm.xyz)**

---

## Overview

Foundational robot models need not only semantic comprehension of task objectives, but also concrete instantiations of intermediate subgoals throughout training and inference. **ACOT-VLA-WM** addresses this by deeply fusing the main ACoT-VLA model with a predictive world model: the world model forecasts multi-view future frames and embeds them as visual subgoals into the main model's training, significantly improving physical fault tolerance and action precision.

### Key Improvements over ACoT-VLA

* **World Model Integration:** A finetuned predictive world model (BAGEL-based) generates multi-view future subgoal images that are embedded into ACoT-VLA training prompts.
* **Mixed Subgoal Sampling (75% / 12.5% / 12.5%):**
  - **75%** — uniformly sample a future frame 0–4 seconds ahead, improving robustness to execution delay and speed perturbations.
  - **12.5%** — use the sub-step terminal frame as subgoal.
  - **12.5%** — use world-model-generated future frames as subgoal.
* **Robust Long-Horizon Execution:** On 5 industrial manipulation scenarios (10 rollouts each), baseline ACoT-VLA achieves **80%** overall success rate, while ACOT-VLA-WM reaches **100%**, including challenging tasks such as picking up a barcode scanner and scanning 5 QR codes on a marble table.

Core training logic lives in `src/openpi/training/subgoal_dataset.py` and `src/openpi/training/sampler.py`, which dynamically schedule world-model predictions and real future frames during training.

---

## Get Started

### 1. Installation

We utilize **uv** to manage the Python environment.

```bash
git clone https://github.com/AgibotTech/ACOT-VLA-WM.git
cd ACOT-VLA-WM
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2. G2 Robot: Video Data → Images（>2× Faster Model Training）

Convert LeRobot video datasets to parquet-embedded image datasets for faster training I/O.

```bash
export HF_LEROBOT_HOME=/data/dataset/Robotdataset/Robotdataset/G2_Robot/phone_packaging

uv run python scripts/predecode_lerobot_videos_to_images.py \
  --input-repo-id video_data \
  --output-repo-id images_data \
  --image-format jpeg \
  --jpeg-quality 98 \
  --num-workers 16 \
  --overwrite
```

### 3. Compute Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py \
  --config-name acot-vla-wm-task \
  --robot-action-dim=24
```

### 4. Model Training

```bash
# Load image-based dataset for training
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run scripts/train.py acot-vla-wm-task \
  --exp-name=id02 \
  --data.use-parquet-images \
  --batch-size 64 \
  --num-workers 16 \
  --overwrite
```

### 5. Model Deployment

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
  --env G2SIM \
  --port 8067 \
  policy:checkpoint \
  --policy.config acot-vla-wm-task \
  --policy.dir "/data/code/ACoT-VLA/checkpoints/acot-vla-wm-task/id01/20000"
```

---


## How to Train World Model?

ACOT-VLA-WM relies on a finetuned predictive world model to generate multi-view future subgoal images. For world model training and finetuning, see the companion repository:

**[BAGEL-WM](https://github.com/alexantaluo0/BAGEL-WM)**



## Acknowledgements

This repo is built upon the [OpenPI](https://github.com/Physical-Intelligence/openpi) framework and the [ACoT-VLA](https://arxiv.org/abs/2601.11404v2) codebase. We sincerely thank the authors for their contributions to the community.
