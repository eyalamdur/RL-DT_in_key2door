import os
import sys
import json
import numpy as np

# Add project root and src to sys.path so imports work
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.models.dt.train_dt import train_dt, load_dt
from src.utils import convert_to_mdp_dataset
from src.env.KeyToDoor import KeyToDoorEnv as k2d
from src.evaluate import evaluate_model


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_optimal_action(env):
    """Get optimal action for current state."""
    if env.room == 0:
        if env.has_key == 0:
            if env.pos[0] > env.key_pos[0]:
                return 0  # up
            elif env.pos[0] < env.key_pos[0]:
                return 1  # down
            elif env.pos[1] < env.key_pos[1]:
                return 3  # right
            elif env.pos[1] > env.key_pos[1]:
                return 2  # left
            else:
                return 4  # pick
        else:
            return 0  # wait
    elif env.room == 1:
        return 0  # wait
    else:  # room 2
        door_pos = [0, env.mid]
        if env.pos[0] > door_pos[0]:
            return 0  # up
        elif env.pos[1] < door_pos[1]:
            return 3  # right
        elif env.pos[1] > door_pos[1]:
            return 2  # left
        else:
            return 0


def collect_trajectory(env, policy='optimal', expert_room=None):
    """Collect a single trajectory with specified policy."""
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    
    while True:
        states.append(obs)
        
        if policy == 'optimal':
            action = get_optimal_action(env)
        elif policy == 'random':
            action = env.action_space.sample()
        elif policy == 'mixed':
            # Use optimal for expert_room, random for others
            if env.room == expert_room:
                action = get_optimal_action(env)
            else:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()
        
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    return {
        'states': states,
        'actions': np.array(actions),
        'rewards': np.array(rewards)
    }


def generate_dataset(env, num_episodes, policy='optimal', filter_wins=False):
    """Generate dataset with specified policy."""
    trajectories = []
    attempts = 0
    max_attempts = num_episodes * 10
    
    while len(trajectories) < num_episodes and attempts < max_attempts:
        if policy == 'mixed':
            expert_room = np.random.randint(0, 3)
            traj = collect_trajectory(env, policy='mixed', expert_room=expert_room)
        else:
            traj = collect_trajectory(env, policy=policy)
        
        total_reward = np.sum(traj['rewards'])
        
        if filter_wins:
            if total_reward > 0:
                trajectories.append(traj)
        else:
            trajectories.append(traj)
        
        attempts += 1
    
    return trajectories


def main():
    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------
    base_dir = os.getcwd()
    quality_datasets_path = os.path.join(base_dir, "results/data")
    dt_models_base_dir = os.path.join(base_dir, "results/models/DT")
    os.makedirs(quality_datasets_path, exist_ok=True)
    
    env = k2d(n=10, render_mode="human")
    
    # ---------------------------------------------------------
    # Generate Quality Datasets (without PPO)
    # ---------------------------------------------------------
    print("\n=== Generate Quality Datasets ===")
    
    # Ground Truth: Optimal policy, only winning trajectories
    gt_path = os.path.join(quality_datasets_path, "ground_truth_10.h5")
    if os.path.exists(gt_path):
        print(f"Ground truth already exists: {gt_path}")
    else:
        print("Generating ground_truth (optimal, winning only)...")
        gt_trajs = generate_dataset(env, num_episodes=1000, policy='optimal', filter_wins=True)
        gt_dataset = convert_to_mdp_dataset(gt_trajs)
        gt_dataset.dump(gt_path)
        print(f"Saved {len(gt_trajs)} trajectories to {gt_path}")
    
    # Half Truth: Mixed policy (1 room optimal, 2 rooms random)
    ht_path = os.path.join(quality_datasets_path, "half_truth_10.h5")
    if os.path.exists(ht_path):
        print(f"Half truth already exists: {ht_path}")
    else:
        print("Generating half_truth (mixed: 1 room optimal, 2 rooms random)...")
        ht_trajs = generate_dataset(env, num_episodes=1000, policy='mixed', filter_wins=False)
        ht_dataset = convert_to_mdp_dataset(ht_trajs)
        ht_dataset.dump(ht_path)
        print(f"Saved {len(ht_trajs)} trajectories to {ht_path}")
    
    # Random: All random actions
    rand_path = os.path.join(quality_datasets_path, "random_10.h5")
    if os.path.exists(rand_path):
        print(f"Random already exists: {rand_path}")
    else:
        print("Generating random (all random actions)...")
        rand_trajs = generate_dataset(env, num_episodes=1000, policy='random', filter_wins=False)
        rand_dataset = convert_to_mdp_dataset(rand_trajs)
        rand_dataset.dump(rand_path)
        print(f"Saved {len(rand_trajs)} trajectories to {rand_path}")
    
    # ---------------------------------------------------------
    # Train Decision Transformers
    # ---------------------------------------------------------
    print("\n=== Decision Transformers Training ===")
    quality_datasets = ["ground_truth_10", "half_truth_10", "random_10"]
    dt_models_list = []

    for quality_dataset in quality_datasets:
        dt_save_path = os.path.join(dt_models_base_dir, f"quality/dt_model_{quality_dataset}.d3")
        os.makedirs(os.path.dirname(dt_save_path), exist_ok=True)
        
        dt_models_list.append([f"dt_{quality_dataset}", dt_save_path])
        
        if os.path.exists(dt_save_path):
            print(f"DT Model already exists: {dt_save_path}")
            continue
        
        dataset_path = os.path.join(quality_datasets_path, f"{quality_dataset}.h5")
        print(f"\n--- Training DT on {quality_dataset} ---")
        
        dt_config = {
            "environment": {
                "grid_size": 10
            },
            "pipeline": {
                "steps": ["train_dt"]
            },
            "dt": {
                "dataset_path": dataset_path,
                "context_size": 10,
                "n_steps": 1000,
                "n_steps_per_epoch": 100,
                "batch_size": 64,
                "learning_rate": 1e-4,
                "save_path": dt_save_path,
                "log_dir": f"logs/quality/dt_{quality_dataset}"
            }
        }
        
        train_dt(dt_config)
        print(f"DT Model saved to: {dt_save_path}")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print("\n=== Evaluation ===")
    eval_config_path = os.path.join(base_dir, "src/experiments/quality/configs/evaluate_quality.json")
    
    eval_config = {
        "grid_size": 10,
        "target_return": 10.0,
        "models": dt_models_list
    }
    save_json(eval_config_path, eval_config)
    
    for model_name, model_path in dt_models_list:
        print(f"\nEvaluating {model_name}...")
        model = load_dt(model_path)
        evaluate_model(env, model, model_type='dt', target_return=10.0)

    print("\nAll evaluations completed successfully!")

if __name__ == "__main__":
    main()
