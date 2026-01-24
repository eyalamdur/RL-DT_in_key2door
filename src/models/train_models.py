import os
# Disable colored logging from d3rlpy/structlog (must be set before importing d3rlpy)
os.environ["NO_COLOR"] = "1"

import logging
import argparse
import sys
import numpy as np
import d3rlpy
import utils
import gymnasium as gym

from models.ppo.train_ppo import train_ppo, load_ppo
from models.dt.train_dt import train_dt
from env.KeyToDoor import KeyToDoorEnv as k2d

# Configure logging
utils.configure_logging()

def run_train_ppo(env, config):
    logging.info("Starting PPO Training...")
    ppo_config = config.get("ppo", {})
    model = train_ppo(env, config=ppo_config.copy())
    return model

def run_collect_data(env: gym.Env, config: dict) -> list[dict]:
    """
    Collect data from the environment.
    Args:
        env: The environment to collect data from.
        config: The configuration dictionary.
    Returns:
        trajs (list[dict]): The collected trajectories.
    """
    logging.info("Starting Data Collection...")
    
    date_str, time_str = utils.get_current_date_time_strings()
    # Default to the best model saved by EvalCallback
    default_ppo_path = f"results/models/PPO/{date_str}/{time_str}/best_model.zip"
    
    # Load PPO model
    ppo_path = config.get("pipeline", {}).get("ppo_model_path", default_ppo_path)
    if os.path.exists(ppo_path):
        model = load_ppo(ppo_path)
        logging.info(f"Loaded PPO model from {ppo_path}")
    else:
        logging.warning(f"PPO model not found at {ppo_path}. Using random agent.")
        model = None

    dc_config = config.get("data_collection", {})
    num_episodes = dc_config.get("num_episodes", 100)
    
    default_save_path = f"results/data/{date_str}/{time_str}/expert_data.h5"
    save_path = dc_config.get("save_path", default_save_path)
    
    trajs = utils.collect_trajectories(env, model, num_episodes=num_episodes, max_traj_length=30)
    dataset = convert_to_mdp_dataset(trajs) # Obs dim 4, act dim 1 for KeyToDoor
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.dump(save_path)
    logging.info(f"Saved dataset to {save_path}")

def convert_to_mdp_dataset(trajectories: list[dict]) -> d3rlpy.dataset.MDPDataset:
    """
    Convert trajectories to d3rlpy MDPDataset.
    Args:
        trajectories: The trajectories to convert.
    Returns:
        dataset (d3rlpy.dataset.MDPDataset)
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []

    for traj in trajectories:
        # Flatten observation if it's a dictionary (KeyToDoor specific)
        obs = traj['states']
        if len(obs) > 0 and (isinstance(obs[0], dict) or (obs.dtype == 'O')):
             flat_obs = []
             for s in obs:
                 flat_s = np.concatenate([
                     np.array([s['room']]), 
                     s['pos'], 
                     np.array([s['has_key']])
                 ])
                 flat_obs.append(flat_s)
             obs = np.array(flat_obs)
        
        observations.append(obs)
        actions.append(traj['actions'])
        rewards.append(traj['rewards'])
        
        curr_terminals = np.zeros(len(traj['actions']))
        curr_terminals[-1] = 1.0 
        terminals.append(curr_terminals)
        curr_timeouts = np.zeros(len(traj['actions']))
        timeouts.append(curr_timeouts)

    observations = np.concatenate(observations)
    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)
    terminals = np.concatenate(terminals)
    timeouts = np.concatenate(timeouts)
    
    if len(actions.shape) == 1:
        actions = actions.reshape(-1, 1)

    return d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

def generate_quality_datasets(env: gym.Env, config: dict):
    logging.info("Starting Quality Dataset Generation...")
    
    date_str, time_str = utils.get_current_date_time_strings()
    
    # Load PPO Model (Expert)
    default_ppo_path = f"results/models/PPO/{date_str}/{time_str}/best_model.zip"
    ppo_path = config.get("pipeline", {}).get("ppo_model_path", default_ppo_path)
    if os.path.exists(ppo_path):
        model = load_ppo(ppo_path)
        logging.info(f"Loaded PPO model from {ppo_path}")
    else:
        logging.error(f"PPO model not found at {ppo_path}. Cannot generate expert data.")
        return

    qc_config = config.get("quality_generation", {})
    num_episodes = qc_config.get("num_episodes", 100)
    base_save_path = qc_config.get("save_path", f"results/data/{date_str}/{time_str}/")
    os.makedirs(base_save_path, exist_ok=True)


    # Collect & Filter Ground Truth (Winning Trajectories)
    logging.info("Collecting Expert Data...")
    
    # Collect more to ensure we have enough winning ones
    raw_trajs = utils.collect_trajectories(env, model, num_episodes=num_episodes * 2, max_traj_length=100)
    
    # Filter for success (reward > 0)
    winning_trajs = [t for t in raw_trajs if np.sum(t['rewards']) > 0]
    
    # Limit to requested number
    if len(winning_trajs) > num_episodes:
        winning_trajs = winning_trajs[:num_episodes]
    
    logging.info(f"Found {len(winning_trajs)} winning trajectories out of {len(raw_trajs)} collected.")
    
    # Save Ground Truth
    gt_path = os.path.join(base_save_path, "ground_truth.h5")
    gt_dataset = convert_to_mdp_dataset(winning_trajs)
    gt_dataset.dump(gt_path)
    logging.info(f"Saved Ground Truth dataset to {gt_path}")


    # Create Half-Truth (50% Correct + 50% Random)
    # ---------------------------------------------------------
    logging.info("Generating Half-Truth Data (1 Room Expert, 2 Random)...")
    half_trajs = []
    
    # We generate trajectories from scratch using mixed policy
    for _ in trange(num_episodes, desc="Generating Half-Truth Data"):
        obs, _ = env.reset()
        states, actions, rewards = [], [], []
        
        # Randomly choose which room (0, 1, 2) will be Expert
        expert_room = np.random.randint(0, 3)
        
        done = False
        while not done:
            # obs is a dict, but we need to access 'room' carefully
            current_room = int(obs['room']) if isinstance(obs, dict) else 0
            
            if current_room == expert_room and model is not None:
                # Use Expert
                action, _ = model.predict(obs, deterministic=False)
            else:
                # Use Random
                action = env.action_space.sample()
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            done = terminated or truncated
            
            # Safety break
            if len(states) >= 1000:
                break
                
        # Calculate RTG
        rtgs = np.cumsum(np.array(rewards)[::-1])[::-1]
        
        half_trajs.append({
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "rtgs": rtgs
        })

    # Save Half-Truth
    ht_path = os.path.join(base_save_path, "half_truth.h5")
    ht_dataset = convert_to_mdp_dataset(half_trajs)
    ht_dataset.dump(ht_path)
    logging.info(f"Saved Half-Truth dataset to {ht_path}")


    # Collect Random (Failures)
    logging.info("Collecting Random Data...")
    # Force random model
    random_trajs = utils.collect_trajectories(env, None, num_episodes=num_episodes, max_traj_length=100)
    
    # Save Random
    rnd_path = os.path.join(base_save_path, "random.h5")
    rnd_dataset = convert_to_mdp_dataset(random_trajs)
    rnd_dataset.dump(rnd_path)
    logging.info(f"Saved Random dataset to {rnd_path}")


def main():
    config = utils.get_config_from_args(
        description="Train Models Pipeline (PPO, Data Collection, DT)",
        default_config_path="src/config/train_config.json"
    )
    
    logging.info(f"Training Configuration Loaded.")
    
    # Setup Environment
    env_config = config.get("environment", {})
    grid_size = env_config.get("grid_size", 4)
    env = k2d(n=grid_size)
    logging.info(f"Environment created (grid_size={grid_size})")

    pipeline_steps = config.get("pipeline", {}).get("steps", [])
    
    if "train_ppo" in pipeline_steps:
        run_train_ppo(env, config)
        
    if "collect_data" in pipeline_steps:
        run_collect_data(env, config)
        
    if "train_dt" in pipeline_steps:
        train_dt(config)

    if "generate_quality_datasets" in pipeline_steps:
        generate_quality_datasets(env, config)

if __name__ == "__main__":
    main()
