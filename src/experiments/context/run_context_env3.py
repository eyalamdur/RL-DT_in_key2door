import os
import sys
import json
import numpy as np
# Add project root and src to sys.path so imports work
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.models.dt.train_dt import train_dt, load_dt
from src.env.KeyToDoor_context import KeyToDoorEnvContext
from src.evaluate import evaluate_model
from src.utils import convert_to_mdp_dataset

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


def collect_trajectory(env):
    """Collect a single trajectory with specified policy."""
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    
    while True:
        states.append(obs)
        action = get_optimal_action(env)
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



def generate_dataset(env, num_episodes):
    """Generate dataset with specified policy."""
    trajectories = []
    attempts = 0
    max_attempts = num_episodes * 10
    
    while len(trajectories) < num_episodes and attempts < max_attempts:
        
        traj = collect_trajectory(env)
        trajectories.append(traj)
        attempts += 1
    
    return trajectories

def main():
    # ---------------------------------------------------------
    # Setup Paths
    # ---------------------------------------------------------
    base_dir = os.getcwd() # Run from project root
    
    # Use existing ground truth trajectories
    data_path = os.path.join(base_dir, "results/data/context_env3/ground_truth_10.h5")
    
    # Setup Environment (size 10)
    env = KeyToDoorEnvContext(n=10, render_mode="human")
    
    # ---------------------------------------------------------
    # Verify Data Exists
    # ---------------------------------------------------------
    print("\n=== Data Verification ===")
    if os.path.exists(data_path):
        print(f"Using existing trajectories: {data_path}")
    else:
        print("Generating ground truth trajectories...")
        trajectories = generate_dataset(env, num_episodes=1000)
        dataset = convert_to_mdp_dataset(trajectories, has_key=False)
        dataset.dump(data_path)
        print(f"Saved {len(trajectories)} trajectories to {data_path}")
    
    # ---------------------------------------------------------
    # Train Decision Transformers (Context Sizes: 10, 30, 60)
    # ---------------------------------------------------------
    print("\n=== Decision Transformers Training ===")
    context_sizes = [10, 30, 60]
    dt_models_list = []

    for ctx in context_sizes:
        dt_save_path = os.path.join(base_dir, f"results/models/DT/context_env3/dt_model_ctx{ctx}.d3")
        dt_models_list.append([f"dt_ctx{ctx}", dt_save_path])
        
        if os.path.exists(dt_save_path):
            print(f"DT Model already exists at: {dt_save_path}")
            continue
        
        print(f"\n--- Training DT with context_size={ctx} ---")
        dt_config = {
            "environment": {
                "grid_size": 10
            },
            "pipeline": {
                "steps": ["train_dt"]
            },
            "dt": {
                "dataset_path": data_path,
                "context_size": ctx,
                "n_steps": 1000,
                "n_steps_per_epoch": 100,
                "batch_size": 64,
                "learning_rate": 1e-4,
                "save_path": dt_save_path,
                "log_dir": f"logs/context_env3/dt_ctx{ctx}"
            }
        }
        
        train_dt(dt_config)
        print(f"DT Model trained and saved to: {dt_save_path}")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print("\n=== Evaluation ===")
    eval_config_path = os.path.join(base_dir, "src/experiments/context/configs/evaluate_context_env3.json")
    
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
