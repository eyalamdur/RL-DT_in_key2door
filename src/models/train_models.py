from models.ppo import train_ppo
from env.KeyToDoor import KeyToDoorEnv as k2d
import utils

def get_models(env) -> list:
    ppo_model = train_ppo.train_ppo(env)
    return [ppo_model]


def main():
    """
    Main function to create the environment and train or load the agents.
    This function initializes the environment, checks if the models are available,
    and trains or loads the PPO and TD3 agents accordingly.
    """
    # Create the environment
    env = k2d()
    utils.color_print("Environment created successfully.")

    get_models(env)
    utils.color_print("Models loaded or trained successfully.", color='green')

if __name__ == "__main__":
    main()