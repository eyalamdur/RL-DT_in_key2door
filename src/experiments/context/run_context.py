import os
import sys
import json

# Add project root and src to sys.path so imports work
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.models.ppo.train_ppo import load_ppo
from src.models.dt.train_dt import train_dt, load_dt
from src.utils import collect_trajectories, convert_to_mdp_dataset
from src.env.KeyToDoor import KeyToDoorEnv as k2d
from src.evaluate import evaluate_model


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def main():
    # ---------------------------------------------------------
    # Setup Paths
    base_dir = os.getcwd() # Run from project root
    
    # Define paths
    data_path = os.path.join(base_dir, f"results/data/trajectories_20.h5")
    
    # Setup Environment
    env = k2d(n=20, render_mode="human")
    
    # ---------------------------------------------------------
    # Load PPO Model
    # ---------------------------------------------------------
    print("\n=== Load PPO Model ===")
    model = load_ppo(os.path.join(base_dir, "results/models/PPO/ppo_room_size_20.zip"))
    print(f"PPO Model loaded from: {os.path.join(base_dir, 'results/models/PPO/ppo_room_size_20.zip')}")
    
    # ---------------------------------------------------------
    # Data Collection
    # ---------------------------------------------------------
    print("\n=== Data Collection ===")
    data_path = os.path.join(base_dir, "results/data/room_size_20.h5")
    
    if os.path.exists(data_path):
        print(f"Trajectories already exist at: {data_path}")
    else:
        print("\n--- Collecting trajectories ---")
        trajectories = collect_trajectories(env, model, num_episodes=100, max_traj_length=120)
        dataset = convert_to_mdp_dataset(trajectories)
        dataset.dump(data_path)
        print(f"Trajectories saved to: {data_path}")
    
    # ---------------------------------------------------------
    # Train Decision Transformers (Context Sizes: 20, 60, 120)
    # ---------------------------------------------------------
    print("\n=== Decision Transformers Training ===")
    context_sizes = [20, 60, 120]
    dt_models_list = []

    for ctx in context_sizes:
        dt_save_path = f"results/models/DT/context/dt_model_ctx{ctx}.d3"
        dt_models_list.append([f"dt_{ctx}", dt_save_path])
        print("path: ", dt_save_path)
        if os.path.exists(dt_save_path):
            print(f"DT Model already exists at: {dt_save_path}")
            continue
        
        dt_config = {
            "environment": {
                "grid_size": 20
            },
            "pipeline": {
                "steps": ["train_dt"]
            },
            "dt": {
                "dataset_path": data_path,
                "context_size": ctx,
                "n_steps": 10000,
                "n_steps_per_epoch": 1000,
                "batch_size": 64,
                "learning_rate": 1e-4,
                "save_path": dt_save_path,
                "log_dir": f"logs/context/dt_ctx{ctx}"
            }
        }
        
        train_dt(dt_config)
        print(f"DT Model trained and saved to: {dt_save_path}")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print("\n=== Evaluation ===")
    eval_config_path = os.path.join(base_dir, "src/experiments/context/configs/evaluate_context.json")
    
    eval_config = {
        "grid_size": 20,
        "target_return": 10.0,
        "models": dt_models_list
    }
    save_json(eval_config_path, eval_config)
    
    for model_type, model_path in dt_models_list:
        print(f"Evaluating {model_type}")
        model = load_dt(model_path)
        evaluate_model(env, model, model_type='dt', target_return=10.0)

    print("\nAll evaluations completed successfully!")

if __name__ == "__main__":
    main()
