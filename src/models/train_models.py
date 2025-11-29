import logging
import utils
from models.ppo.train_ppo import train_ppo
from env.KeyToDoor import KeyToDoorEnv as k2d

# Configure logging
utils.configure_logging()

def add_training_args(parser):
    """Add training-specific arguments to the parser."""
    # Environment parameters
    parser.add_argument('--grid_size', type=int,
                        help='Grid size (n) for the environment')

    # Training parameters
    parser.add_argument('--policy', type=str,
                        help='Policy type (e.g. MultiInputPolicy)')
    parser.add_argument('--device', type=str,
                        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate')
    parser.add_argument('--n_steps', type=int,
                        help='Number of steps to run for each environment per update')
    parser.add_argument('--batch_size', type=int,
                        help='Minibatch size')
    parser.add_argument('--n_epochs', type=int,
                        help='Number of epoch when optimizing the surrogate loss')
    parser.add_argument('--gamma', type=float,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float,
                        help='Factor for trade-off of bias vs variance for GAE')
    parser.add_argument('--ent_coef', type=float,
                        help='Entropy coefficient for the loss calculation')
    parser.add_argument('--verbose', type=int,
                        help='Verbosity level')
    parser.add_argument('--total_timesteps', type=int,
                        help='Total timesteps for training')

def main():
    """
    Main function to create the environment and train the agents.
    """
    # Parse arguments (flat merge of config and CLI args at top level if config is flat)
    # With nested config, 'env' and 'ppo' keys will exist in config dict.
    config = utils.get_config_from_args(
        description='Train PPO agent for KeyToDoor environment',
        default_config_path="src/config/ppo_config.json",
        custom_args_setup=add_training_args
    )
    
    # Extract Environment Configuration
    env_config = config.get("env", {})
    # grid_size can be in env_config, or at top level (CLI override)
    grid_size = config.get("grid_size") or env_config.get("grid_size", 4)
    
    # Create the environment
    env = k2d(n=grid_size)
    logging.info(f"Environment created successfully with grid_size={grid_size}.")

    # Extract PPO Configuration
    ppo_config = config.get("ppo", {}).copy()
    
    # Update ppo_config with any top-level CLI arguments that match PPO parameters
    # This assumes utils.get_config_from_args put CLI args at the top level
    ppo_param_names = [
        "policy", "device", "learning_rate", "n_steps", "batch_size", 
        "n_epochs", "gamma", "gae_lambda", "ent_coef", "verbose", "total_timesteps"
    ]
    
    for param in ppo_param_names:
        if param in config:
            ppo_config[param] = config[param]

    # Train PPO
    logging.info(f"Training PPO with config: {ppo_config}")
    train_ppo(env, config=ppo_config)
    logging.info("Models trained successfully.")

if __name__ == "__main__":
    main()
