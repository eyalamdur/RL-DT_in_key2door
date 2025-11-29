import logging
import json
import gymnasium as gym
from stable_baselines3 import PPO
from env.KeyToDoor import KeyToDoorEnv as k2d
from models.ppo.train_ppo import load_ppo
import utils

# Configure logging
utils.configure_logging()

def add_eval_args(parser):
    """Add evaluation-specific arguments to the parser."""
    # Model overrides (optional)
    parser.add_argument('--model_path', type=str,
                        help='Path to a single model to evaluate (overrides config models list)')
    parser.add_argument('--grid_size', type=int,
                        help='Grid size (n) for the environment')

def evaluate_model(env: gym.Env, model):

    # Initialize evaluate params
    state, _ = env.reset()
    done = False
    cumulative_reword = 0

    # run the test using env.step and sum the rewards
    while not done:
        env.render()
        action = model.predict(state)[0]
        state, reward, terminated, truncated, _ = env.step(action)
        env.print_action(action)
        cumulative_reword += reward
        if terminated or truncated:
            done = True

    logging.info(f"agent's cumulative_reward: {cumulative_reword}")


def main():
    
    config = utils.get_config_from_args(
        description='Evaluate trained agents',
        default_config_path="src/config/evaluate_config.json",
        custom_args_setup=add_eval_args
    )
    
    # Initialize environment with human rendering
    grid_size = config.get("grid_size", 4)
    env = k2d(render_mode="human", n=grid_size)
    
    # Post-process config for evaluation-specific logic
    if 'model_path' in config:
        config['models'] = [["custom_model", config['model_path']]]

    models_list = config.get("models", [])
    
    if not models_list:
        logging.warning("No models to evaluate. Please provide models in config or use --model_path")
        return

    for model_name, model_path in models_list:
        logging.info(f"==================================================")
        logging.info(f"Evaluating Model: {model_name}")
        logging.info(f"Model Path: {model_path}")
        logging.info(f"Grid Size: {grid_size}")
        logging.info(f"==================================================")
        
        try:
            model = load_ppo(model_path)
            evaluate_model(env, model)
        except Exception as e:
            logging.error(f"Failed to load or evaluate model {model_name}: {e}")

if __name__ == "__main__":
    main()
