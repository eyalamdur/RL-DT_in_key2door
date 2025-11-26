import json
import os
import argparse
import logging
import sys
from .ppo.train_ppo import train_ppo
from ..env.KeyToDoor import KeyToDoorEnv as k2d

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def parse_args():
    """
    Parse command line arguments with support for config files.
    Either provide all arguments or use --config with a JSON file.
    If both are provided, the provided command line arguments will override the config file ones.
    """
    parser = argparse.ArgumentParser(description='Train PPO agent for KeyToDoor environment')
    
    # Configuration file option
    parser.add_argument('--config', type=str, default="src/config/ppo_config.json",
                        help='Path to JSON configuration file (overrides all other arguments)')
    
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

    args = parser.parse_args()
    
    # If config file is provided (default or explicit), load it and override args
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        
        # Override args with config values, but giving priority to command line arguments
        final_config = config.copy()
        
        # Override config with command line arguments that are not None
        for key, value in args.__dict__.items():
            if value is not None and key != 'config':
                final_config[key] = value
                
        return final_config
    else:
        # If no config file, just use args (filtering out None)
        final_config = {k: v for k, v in args.__dict__.items() if v is not None and k != 'config'}
        return final_config


def main():
    """
    Main function to create the environment and train the agents.
    """
    # Parse arguments
    config = parse_args()
    
    # Create the environment
    env = k2d()
    logging.info("Environment created successfully.")

    # Train PPO
    logging.info(f"Training PPO with config: {config}")
    train_ppo(env, config=config)
    logging.info("Models trained successfully.")

if __name__ == "__main__":
    main()
