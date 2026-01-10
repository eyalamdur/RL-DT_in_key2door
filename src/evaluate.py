import logging
import numpy as np
import torch
import gymnasium as gym
import d3rlpy
from d3rlpy.algos.transformer.inputs import TransformerInput
from stable_baselines3 import PPO
from env.KeyToDoor import KeyToDoorEnv as k2d
from models.ppo.train_ppo import load_ppo
from models.dt.train_dt import load_dt
import utils
import os

# Configure logging
utils.configure_logging()

def add_eval_args(parser):
    """Add evaluation-specific arguments to the parser."""
    parser.add_argument('--model_path', type=str,
                        help='Path to a single model to evaluate (overrides config list)')
    parser.add_argument('--model_type', type=str, choices=['ppo', 'dt'],
                        help='Type of model: ppo (SB3) or dt (d3rlpy). Required if model_path is provided.')
    parser.add_argument('--target_return', type=float, default=10.0,
                        help='Target return for Decision Transformer')

def flatten_state(s):
    """Flatten dict state to numpy array for DT."""
    return np.concatenate([
        np.array([s['room']], dtype=np.float32), 
        s['pos'].astype(np.float32), 
        np.array([s['has_key']], dtype=np.float32)
    ])

def evaluate_ppo(env: gym.Env, model):
    """Evaluate a PPO model."""
    state, _ = env.reset()
    done = False
    cumulative_reward = 0
    
    # Print initial state
    env.render()
    
    while not done:
        action, _ = model.predict(state)
        state, reward, terminated, truncated, _ = env.step(action)
        env.print_action(action)
        env.render()
        cumulative_reward += reward
        done = terminated or truncated

    logging.info(f"PPO agent's cumulative_reward: {cumulative_reward}")
    return cumulative_reward

def evaluate_dt(env: gym.Env, model, target_return=10.0, context_size=20):
    """Evaluate a Decision Transformer model using d3rlpy v2.x API."""
    state, _ = env.reset()
    done = False
    cumulative_reward = 0
    
    # Print initial state
    env.render()
    
    # Initialize trajectory context
    observations = []
    actions = []
    rewards = []
    returns_to_go = []
    timesteps = []
    
    step = 0
    current_rtg = target_return
    
    while not done:
        flat_state = flatten_state(state)
        observations.append(flat_state)
        returns_to_go.append(current_rtg)
        timesteps.append(step)
        
        # Build TransformerInput from trajectory context
        # Use only the last `context_size` steps
        ctx_start = max(0, len(observations) - context_size)
        ctx_len = len(observations) - ctx_start
        
        # observations: (L, obs_dim)
        obs_array = np.array(observations[ctx_start:], dtype=np.float32)
        
        # returns_to_go: (L, 1)
        rtg_array = np.array(returns_to_go[ctx_start:], dtype=np.float32)[:, np.newaxis]
        
        # timesteps: (L,)
        ts_array = np.array(timesteps[ctx_start:], dtype=np.int32)
        
        # rewards: (L, 1) - must match observations length
        # Pad with 0 for current step (reward not received yet)
        padded_rewards = rewards[ctx_start:] + [0.0]  # Add dummy for current step
        rew_array = np.array(padded_rewards, dtype=np.float32)[:, np.newaxis]
        
        # Actions: (L,) - must match observations length
        # Pad with dummy action (0) for current observation (action not taken yet)
        # Ensure all actions are proper ints within valid range (0-4)
        padded_actions = [int(a) % 5 for a in actions[ctx_start:]] + [0]
        act_array = np.array(padded_actions, dtype=np.int32)
        
        # Create TransformerInput
        inpt = TransformerInput(
            observations=obs_array,
            actions=act_array,
            rewards=rew_array,
            returns_to_go=rtg_array,
            timesteps=ts_array,
        )
        
        # Predict action - DT returns logits, use argmax to get discrete action
        raw_logits = model.predict(inpt)
        action = int(np.argmax(raw_logits))
        actions.append(action)
        
        # Take step
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        env.print_action(action)
        env.render()
        
        cumulative_reward += reward
        current_rtg = target_return - cumulative_reward
        step += 1
        done = terminated or truncated

    logging.info(f"DT agent's cumulative_reward: {cumulative_reward}")
    return cumulative_reward

def evaluate_model(env: gym.Env, model, model_type='ppo', target_return=10.0):
    """Dispatch to appropriate evaluation function."""
    if model_type == 'ppo':
        return evaluate_ppo(env, model)
    elif model_type == 'dt':
        return evaluate_dt(env, model, target_return=target_return)
    else:
        logging.error(f"Unknown model type: {model_type}")

def main():
    config = utils.get_config_from_args(
        description='Evaluate trained agents',
        default_config_path="src/config/evaluate_config.json",
        custom_args_setup=add_eval_args
    )
    
    grid_size = config.get("grid_size", 4)
    env = k2d(render_mode="human", n=grid_size)
    
    # Check for single model override via CLI
    cli_model_path = config.get('model_path')
    cli_model_type = config.get('model_type')
    
    models_to_evaluate = []
    
    if cli_model_path:
        if not cli_model_type:
             logging.warning("When specifying --model_path, please also specify --model_type (ppo/dt)")
             # default to ppo
             cli_model_type = 'ppo'
        models_to_evaluate.append((cli_model_type, cli_model_path))
    else:
        # Load from config list
        models_list = config.get('models', [])
        models_to_evaluate = models_list # Format: [[type, path], ...]

    target_return = config.get('target_return', 10.0)

    if not models_to_evaluate:
        logging.warning("No models found to evaluate. Check config or provide arguments.")
        return

    for model_type, model_path in models_to_evaluate:
        logging.info(f"--- Evaluating {model_type.upper()} model from {model_path} ---")
        
        # Check if file exists (with extension or without)
        if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip") and not os.path.exists(model_path + ".d3"):
             logging.warning(f"Model file not found: {model_path}")
             continue

        try:
            # Handle both 'ppo' and 'ppo_model' as PPO type
            if model_type in ['ppo', 'ppo_model']:
                model = load_ppo(model_path)
                model_type_for_eval = 'ppo'
            else:
                model = load_dt(model_path)
                model_type_for_eval = 'dt'
            evaluate_model(env, model, model_type_for_eval, target_return)
            
        except Exception as e:
            logging.error(f"Failed evaluation for {model_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
