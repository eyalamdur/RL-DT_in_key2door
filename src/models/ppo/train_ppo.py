import gymnasium as gym
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import utils

MAX_EPISODE_STEPS = 30  # Maximum steps per episode for PPO in k2d environment

def train_ppo(env: gym.Env, config: dict = None, num_episodes: int = 1000) -> PPO:
    """
    Train a PPO agent on the given environment.
    Args:
        env (gym.Env): The environment to train the agent on.
        config (dict): The configuration for PPO training.
        num_episodes (int): The number of episodes to train the agent.
    Returns:
        agent (PPO): The trained PPO agent.
    """
    if config is None:
        raise ValueError("Config is required")

    # Extract non-PPO args if present
    total_timesteps = config.pop("total_timesteps", num_episodes * MAX_EPISODE_STEPS)
    
    # Create evaluation callback to save best model
    # Note: Ideally we should use a separate evaluation environment
    # For now, we reuse the training env or assume it handles reset correctly
    eval_callback = create_eval_callback(env, total_timesteps)

    # Create the PPO agent
    model = PPO(env=env, **config)
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps, 
        reset_num_timesteps=False, 
        progress_bar=True,
        callback=eval_callback
    )

    return model

def load_ppo(model_path: str) -> PPO:
    """
    Load a trained PPO agent from a file.
    Args:
        model_path (str): The path to the saved PPO model.
    Returns:
        agent (PPO): The loaded PPO agent.
    """
    model = PPO.load(model_path, device="cpu")
    return model

def create_eval_callback(eval_env: gym.Env, total_timesteps: int) -> EvalCallback:
    """
    Create an evaluation callback to save the best model.
    Args:
        eval_env: The environment to evaluate on.
        total_timesteps: The total number of timesteps to train.
    Returns:
        eval_callback (EvalCallback)
    """
    date_str, time_str = utils.get_current_date_time_strings()
    best_model_path = f"results/models/PPO/{date_str}/{time_str}/"
    
    # Evaluate every 5% of training
    eval_freq = max(1, int(total_timesteps / 20))
    
    return EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=best_model_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
