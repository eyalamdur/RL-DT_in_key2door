import os
import pickle
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import trange
from typing import List, Dict, Optional
from stable_baselines3.common.base_class import BaseAlgorithm  # parent of PPO, TD3, etc.
import logging
import sys
import json
import argparse

# -------------------------------------------- Logging & Config Utilities -------------------------------------------- #
def configure_logging(level=logging.INFO):
    """
    Configure logging to stream to stdout.
    This ensures SLURM captures logs in .out files instead of .err files.
    """
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True # Force reconfiguration if already configured
    )

def get_current_date_time_strings() -> tuple[str, str]:
    """
    Get current date and time strings formatted for file paths.
    Returns:
        tuple: (date_str, time_str) where date_str is DD/MM/YYYY and time_str is HH_MM_SS
    """
    now = datetime.now()
    date_str = now.strftime("%d_%m_%Y")
    time_str = now.strftime("%H:%M:%S")
    return date_str, time_str

def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file with date substitution.
    Args:
        config_path: The path to the configuration file.
    Returns:
        config (dict): The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Date substitution
        date_str, time_str = get_current_date_time_strings()
        
        def recursive_date_sub(item):
            if isinstance(item, str):
                item = item.replace("{DATE}", date_str)
                item = item.replace("{TIME}", time_str)
                return item
            elif isinstance(item, dict):
                return {k: recursive_date_sub(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [recursive_date_sub(i) for i in item]
            else:
                return item
                
        return recursive_date_sub(config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def get_config_from_args(description="K2D Script", default_config_path="src/config/config.json", custom_args_setup=None):
    """
    Parse command line arguments and load configuration.
    
    Args:
        description (str): Description for argparse.
        default_config_path (str): Default path for configuration file.
        custom_args_setup (callable): Function to add custom arguments to parser.
                                      Signature: custom_args_setup(parser)
    
    Returns:
        dict: The final configuration dictionary.
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Configuration file option
    parser.add_argument('--config', type=str, default=default_config_path,
                        help='Path to JSON configuration file (overrides all other arguments)')
    
    # Allow adding custom arguments
    if custom_args_setup:
        custom_args_setup(parser)
    
    args = parser.parse_args()
    
    # Load config file if present
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        logging.info(f"Loaded config from {args.config}")
    else:
        logging.warning(f"Config file not found or not provided: {args.config}. Using defaults/CLI args only.")
    
    # Override config with CLI arguments (excluding 'config' itself)
    final_config = config.copy()
    for key, value in args.__dict__.items():
        if value is not None and key != 'config':
            final_config[key] = value
            
    return final_config

# -------------------------------------------- Environment Utilities -------------------------------------------- #
def collect_trajectories(
    env: gym.Env,
    model: Optional[BaseAlgorithm] = None,
    num_episodes: int = 10,
    min_traj_length: int = 1,
    max_traj_length: int = 100,
    deterministic: bool = True
) -> List[Dict[str, np.ndarray]]:
    """
    Collect trajectories using a model or random actions. Stores return-to-go (RTG) instead of rewards.
    Args:
        env (gym.Env): The environment.
        model (BaseAlgorithm or None): PPO, TD3, or None for random actions.
        num_episodes (int): Number of episodes to collect.
        min_traj_length (int): Minimum length to keep a trajectory.
        deterministic (bool): If using model, whether actions are deterministic.
    Returns:
        List[Dict]: Each dict contains 'states', 'actions', 'rtgs' for one episode.
    """
    trajectories = []
    model_name = type(model).__name__ if model is not None else "Random"

    for episode in trange(1, num_episodes + 1, desc=f"Collecting trajectories from agent {model_name}"):
        obs, _ = env.reset()
        states, actions, rewards = [], [], []

        for _ in range(max_traj_length): 
            # Use model to predict action or sample from action space if no model is provided
            action = model.predict(obs, deterministic=deterministic)[0] if model else env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            
            if terminated or truncated:
                break

        if len(states) >= min_traj_length:
            rtgs = np.cumsum(rewards[::-1])[::-1]  # Return-to-go at each timestep
            trajectories.append({
                "states": np.array(states),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "rtgs": rtgs
            })

    return trajectories
