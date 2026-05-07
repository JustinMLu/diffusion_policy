# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diffusion Policy: visuomotor policy learning via action diffusion for robot manipulation. Implements multiple policy methods (diffusion, BET, IBC, robomimic) across multiple task environments (Push-T, Block Pushing, Kitchen, Robomimic suite).

## Environment Setup

```bash
# Conda environment (name: robodiff), Python 3.9, PyTorch 1.12.1, CUDA 11.6
mamba env create -f conda_environment.yaml   # or conda
conda activate robodiff

# Package installed in dev mode
pip install -e .

# System deps (Ubuntu, for mujoco)
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

## Key Commands

```bash
# Training (uses Hydra for config)
python train.py --config-name=train_diffusion_unet_hybrid_workspace task=pusht_image training.seed=42 training.device=cuda:0

# Training with custom output dir
python train.py --config-dir=. --config-name=<config>.yaml hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# Evaluation from checkpoint
python eval.py --checkpoint <path>.ckpt --output_dir <dir> --device cuda:0

# Multi-seed training via Ray
ray start --head --num-gpus=3
python ray_train_multirun.py --config-name=<config>.yaml --seeds=42,43,44 --monitor_key=test/mean_score

# Tests
python -m pytest tests/
python -m pytest tests/test_replay_buffer.py
```

## Architecture

The codebase is designed so that N tasks and M methods require O(N+M) code, not O(N*M). Tasks and methods are independent; they communicate through a unified observation/action interface.

### Task Side (per environment)
- **Dataset** (`diffusion_policy/dataset/`): adapts data to the interface, returns `{obs, action}` dicts. Uses `ReplayBuffer` (zarr-backed) + `SequenceSampler` for episode-aware sampling with correct padding.
- **EnvRunner** (`diffusion_policy/env_runner/`): runs a `Policy` in the environment, returns logs/metrics for wandb.
- **Config** (`diffusion_policy/config/task/<name>.yaml`): defines `shape_meta`, `dataset._target_`, and `env_runner._target_`.

### Method Side (per algorithm)
- **Policy** (`diffusion_policy/policy/`): implements `predict_action(obs_dict) -> action_dict` and `compute_loss(batch)`. Handles normalization internally via `set_normalizer`.
- **Workspace** (`diffusion_policy/workspace/`): manages the full training/eval lifecycle. All training state as object attributes is auto-checkpointed.
- **Config** (`diffusion_policy/config/train_*_workspace.yaml`): top-level config with `_target_` pointing to workspace class.

### Observation/Action Interface
- **LowDim**: obs `(B, To, Do)` → action `(B, Ta, Da)`
- **Image**: obs dict with mixed types (rgb as `(B, To, H, W, 3)` float32 [0,1], low_dim as `(B, To, *)`) → action `(B, Ta, Da)`
- Terminology: `To`/`n_obs_steps` = observation horizon, `Ta`/`n_action_steps` = action horizon, `T`/`horizon` = prediction horizon

### Configuration System
Hydra with OmegaConf. The `_target_` field specifies which Python class to instantiate. A custom `${eval:'...'}` resolver enables inline Python expressions. Select task via `task=<task_name>` override (loads `config/task/<task_name>.yaml`).

### Key Internals
- **LinearNormalizer** (`model/common/normalizer.py`): saved as part of policy checkpoint, applied on GPU. Normalization bugs are a common source of issues — inspect `scale`/`bias` vectors when debugging.
- **ReplayBuffer** (`common/replay_buffer.py`): zarr-based storage with episodes concatenated along time axis. `meta/episode_ends` tracks episode boundaries.
- **SequenceSampler** (`common/sampler.py`): handles padding at episode boundaries for `To`/`Ta`. Read this before implementing custom sampling.
- **EMA** (`model/diffusion/ema_model.py`): EMA is used instead of BatchNorm (EMA + BatchNorm breaks performance; GroupNorm is used instead).
- **AsyncVectorEnv** (`gym_util/async_vector_env.py`): fork-based parallelism. Environments that create OpenGL contexts at init (e.g., robosuite) need a `dummy_env_fn` workaround to avoid segfaults in child processes.

### Adding a New Task
Imitate `pusht_image_dataset.py`, `pusht_image_runner.py`, and `config/task/pusht_image.yaml`. Ensure `shape_meta` matches input/output shapes and `_target_` fields point to new classes.

### Adding a New Method
Imitate `train_diffusion_unet_image_workspace.py`, `diffusion_unet_image_policy.py`, and `config/train_diffusion_unet_image_workspace.yaml`. Ensure workspace yaml `_target_` points to the new workspace class.

## Data

Training data goes under `data/` at repo root. Datasets use zarr format (`.zarr` directories or `.zarr.zip` files). Download from https://diffusion-policy.cs.columbia.edu/data/training/.

## Logging

Training logs to wandb (project configured in workspace yaml) and local `logs.json.txt`. Evaluation metrics logged as `test/mean_score`. Checkpoints saved to `data/outputs/` with top-k by `test_mean_score`.
