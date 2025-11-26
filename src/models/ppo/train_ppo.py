import gymnasium as gym
# import utils
from tqdm import trange
from stable_baselines3 import PPO

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

    # Create the PPO agent
    model = PPO(env=env,**config)
    steps_per_episode = env.spec.max_episode_steps or MAX_EPISODE_STEPS
    
    # train the model
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=True)

    # model_id = utils.get_next_run_id("results/models/PPO", "models")
    model_id = "1"
    model.save(f"results/models/PPO/ppo_{model_id}")

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