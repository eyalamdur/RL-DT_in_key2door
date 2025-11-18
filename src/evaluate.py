import gymnasium as gym
from stable_baselines3 import PPO
from KeyToDoor import KeyToDoorEnv as k2d
from ppo import train_ppo

def evaluate_model (env: gym.Env, model, max_episode_length = 30):

    # Initialize evaluate params
    state, _ = env.reset()
    done = False
    cumulative_reword = 0

    # run the test using env.step and sum the rewards
    steps = 0
    while not done and steps < max_episode_length:
        action = model.predict(state)[0]
        state, reward, terminated, truncated, _ = env.step(action)
        env.print_action(action)
        env.render()
        cumulative_reword += reward
        steps += 1
        if terminated or truncated:
            done = True

    print("agent's cumulative_reward: ", cumulative_reword)


if __name__ == "__main__":
    env = k2d()
    model_path = "results/models/PPO/ppo_1"
    model = train_ppo.load_ppo(model_path)  # Load the model on CPU
    evaluate_model(env, model)