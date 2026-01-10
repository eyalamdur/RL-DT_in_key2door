# RL-DT_in_key2door

A comparative study of **online Reinforcement Learning (RL)** versus **offline imitation learning** in the Key-to-Door environment. This project trains a PPO agent (Proximal Policy Optimization) as an expert, collects its trajectories, and uses them to train a Decision Transformer (DT) to perform the same task through sequence modeling.

## Overview

### The Key-to-Door Environment 🗝️🚪

The Key-to-Door environment is a sequential decision-making task implemented as a Gymnasium environment. The agent must:
- Navigate through **3 rooms** in a grid-based world (configurable grid size, default 4×4)
- **Pick up a key** in Room 0 at a specific location
- **Reach the door** in Room 2 while holding the key to succeed
- Complete the task within a limited number of steps per room (2n steps where n is grid size)

**Reward Structure:**
- Small penalty per step (-0.01) to encourage efficiency
- Large terminal reward (+10.0) upon successfully reaching the door with the key

### Approaches Compared 🔄

1. **PPO (Proximal Policy Optimization)**: An online RL algorithm that learns through trial-and-error interactions with the environment, implemented using `stable-baselines3`. Acts as the "teacher" agent.

2. **Decision Transformer (DT)**: An offline imitation learning approach that models the problem as sequence prediction. The DT learns from expert trajectories collected from the PPO agent, implemented using `d3rlpy`. Acts as the "student" agent.

### Research Motivation 🔬

This comparison explores whether offline learning methods (DT) can effectively mimic and potentially improve upon online RL agents (PPO) when trained on expert demonstrations, without requiring direct environment interaction during training. 

## Training 🚀

To train the models, use the `train.sh` script which automatically submits to SLURM with timestamped logs:

```bash
./train.sh
```

The training pipeline consists of three steps (configured in `src/config/train_config.json`):
1. **Train PPO**: Train a Proximal Policy Optimization agent using stable-baselines3
2. **Collect Data**: Collect expert trajectories from the trained PPO model
3. **Train DT**: Train a Decision Transformer using the collected data

You can customize the pipeline by editing `src/config/train_config.json`:
- Modify `pipeline.steps` to include/exclude steps: `["train_ppo", "collect_data", "train_dt"]`
- Adjust hyperparameters for PPO, data collection, and DT
- Change paths (uses `{DATE}` and `{TIME}` placeholders for automatic timestamping)

Training outputs are saved to:
- PPO models: `results/models/PPO/{DATE}/{TIME}/best_model/`
- Collected data: `results/data/{DATE}/{TIME}/expert_data.h5`
- DT models: `results/models/DT/{DATE}/{TIME}/dt_model.d3`
- Logs: `logs/train/{DATE}/{TIME}/`

## Evaluation 📈

To evaluate trained models, use the `evaluate.sh` script:

```bash
./evaluate.sh
```

The evaluation script loads models from `src/config/evaluate_config.json`. You can specify models to evaluate:

```json
{
  "models": [
    ["ppo_model", "results/models/PPO/10_01_2026/01:10:35/best_model.zip"],
    ["dt", "results/models/DT/10_01_2026/01:10:35/dt_model.d3"]
  ],
  "grid_size": 4,
  "target_return": 10.0
}
```

Evaluation logs are saved to: `logs/eval/{DATE}/{TIME}/`

You can also evaluate a single model via command line:
```bash
./evaluate.sh --model_path path/to/model.zip --model_type ppo --target_return 10.0
```
