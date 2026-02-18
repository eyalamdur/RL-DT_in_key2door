import os
import sys
import json
import numpy as np

# Add project root and src to sys.path so imports work
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.models.dt.train_dt import train_dt, load_dt
from src.env.KeyToDoor import KeyToDoorEnv as k2d
import d3rlpy
from d3rlpy.algos.transformer.inputs import TransformerInput


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def flatten_state_normalized(s, grid_size):
    """Flatten dict state to normalized numpy array for DT."""
    return np.array([
        s['room'] / 2.0,  # Normalize room to [0, 1]
        s['pos'][0] / (grid_size - 1),  # pos_x normalized
        s['pos'][1] / (grid_size - 1),  # pos_y normalized
        float(s['has_key']),  # 0 or 1
        s['key_pos'][0] / (grid_size - 1),  # key_pos_x normalized
        s['key_pos'][1] / (grid_size - 1),  # key_pos_y normalized
    ], dtype=np.float32)


def evaluate_dt_normalized(env, model, grid_size, target_return=10.0, context_size=20):
    """Evaluate a Decision Transformer model using NORMALIZED observations."""
    state, _ = env.reset()
    done = False
    cumulative_reward = 0
    
    env.render()
    
    observations = []
    actions = []
    rewards = []
    returns_to_go = []
    timesteps = []
    
    step = 0
    current_rtg = target_return
    
    while not done:
        flat_state = flatten_state_normalized(state, grid_size)
        observations.append(flat_state)
        returns_to_go.append(current_rtg)
        timesteps.append(step)
        
        ctx_start = max(0, len(observations) - context_size)
        
        obs_array = np.array(observations[ctx_start:], dtype=np.float32)
        rtg_array = np.array(returns_to_go[ctx_start:], dtype=np.float32)[:, np.newaxis]
        ts_array = np.array(timesteps[ctx_start:], dtype=np.int32)
        
        padded_rewards = rewards[ctx_start:] + [0.0]
        rew_array = np.array(padded_rewards, dtype=np.float32)[:, np.newaxis]
        
        padded_actions = actions[ctx_start:] + [0]
        act_array = np.array(padded_actions, dtype=np.int32)
        
        inpt = TransformerInput(
            observations=obs_array,
            actions=act_array,
            rewards=rew_array,
            returns_to_go=rtg_array,
            timesteps=ts_array,
        )
        
        raw_logits = model.predict(inpt)
        action = int(np.argmax(raw_logits))
        actions.append(action)
        
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        env.print_action(action)
        env.render()
        
        cumulative_reward += reward
        current_rtg = target_return - cumulative_reward
        step += 1
        done = terminated or truncated

    print(f"DT agent's cumulative_reward: {cumulative_reward}")
    return cumulative_reward


def get_optimal_action(env):
    """Get optimal action for current state."""
    if env.room == 0:
        if env.has_key == 0:
            if env.pos[0] > env.key_pos[0]:
                return 0  # up
            elif env.pos[0] < env.key_pos[0]:
                return 1  # down
            elif env.pos[1] < env.key_pos[1]:
                return 3  # right
            elif env.pos[1] > env.key_pos[1]:
                return 2  # left
            else:
                return 4  # pick
        else:
            return 0  # wait
    elif env.room == 1:
        return 0  # wait
    else:  # room 2
        door_pos = [0, env.mid]
        if env.pos[0] > door_pos[0]:
            return 0  # up
        elif env.pos[1] < door_pos[1]:
            return 3  # right
        elif env.pos[1] > door_pos[1]:
            return 2  # left
        else:
            return 0


def collect_trajectory(env, policy='optimal', expert_room=None):
    """Collect a single trajectory with specified policy."""
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    
    while True:
        states.append(obs)
        
        if policy == 'optimal':
            action = get_optimal_action(env)
        elif policy == 'random':
            action = env.action_space.sample()
        elif policy == 'mixed':
            # Use optimal for expert_room, random for others
            if env.room == expert_room:
                action = get_optimal_action(env)
            else:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()
        
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    return {
        'states': states,
        'actions': np.array(actions),
        'rewards': np.array(rewards)
    }


def generate_dataset(env, num_episodes, policy='optimal', filter_wins=False):
    """Generate dataset with specified policy."""
    trajectories = []
    attempts = 0
    max_attempts = num_episodes * 10
    
    while len(trajectories) < num_episodes and attempts < max_attempts:
        if policy == 'mixed':
            expert_room = np.random.randint(0, 3)
            traj = collect_trajectory(env, policy='mixed', expert_room=expert_room)
        else:
            traj = collect_trajectory(env, policy=policy)
        
        total_reward = np.sum(traj['rewards'])
        
        if filter_wins:
            if total_reward > 0:
                trajectories.append(traj)
        else:
            trajectories.append(traj)
        
        attempts += 1
    
    return trajectories


def convert_to_mdp_dataset_normalized(trajectories, grid_size):
    """Convert trajectories to MDPDataset with NORMALIZED observations.
    
    Normalizes positions to [0, 1] range for grid-size invariance.
    Observation format: [room/2, pos_x_norm, pos_y_norm, has_key, key_pos_x_norm, key_pos_y_norm]
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []

    for traj in trajectories:
        obs = traj['states']
        flat_obs = []
        for s in obs:
            # Normalize: room stays 0-2, positions scaled to [0,1]
            flat_s = np.array([
                s['room'] / 2.0,  # Normalize room to [0, 1]
                s['pos'][0] / (grid_size - 1),  # pos_x normalized
                s['pos'][1] / (grid_size - 1),  # pos_y normalized
                float(s['has_key']),  # 0 or 1
                s['key_pos'][0] / (grid_size - 1),  # key_pos_x normalized
                s['key_pos'][1] / (grid_size - 1),  # key_pos_y normalized
            ], dtype=np.float32)
            flat_obs.append(flat_s)
        
        observations.append(np.array(flat_obs))
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


def concatenate_datasets(datasets):
    """Concatenate multiple d3rlpy MDPDatasets into one."""
    import d3rlpy
    import h5py
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_timeouts = []
    
    for dataset in datasets:
        # Load dataset if it's a path string
        if isinstance(dataset, str):
            with h5py.File(dataset, 'r') as f:
                # d3rlpy stores episodes separately
                ep_count = sum(1 for k in f.keys() if k.startswith('observations_'))
                for ep_idx in range(ep_count):
                    all_observations.append(f[f'observations_{ep_idx}'][:])
                    all_actions.append(f[f'actions_{ep_idx}'][:])
                    all_rewards.append(f[f'rewards_{ep_idx}'][:])
                    # Create terminals (1 at end of each episode)
                    term = np.zeros(len(f[f'observations_{ep_idx}'][:]))
                    term[-1] = 1.0
                    all_terminals.append(term)
                    all_timeouts.append(np.zeros(len(term)))
        else:
            # It's already an MDPDataset - extract episodes
            for episode in dataset.episodes:
                all_observations.append(episode.observations)
                all_actions.append(episode.actions)
                all_rewards.append(episode.rewards)
                term = np.zeros(len(episode.observations))
                term[-1] = 1.0
                all_terminals.append(term)
                all_timeouts.append(np.zeros(len(term)))
    
    # Concatenate all
    observations = np.concatenate(all_observations)
    actions = np.concatenate(all_actions)
    rewards = np.concatenate(all_rewards)
    terminals = np.concatenate(all_terminals)
    timeouts = np.concatenate(all_timeouts)
    
    # Ensure actions are 2D
    if len(actions.shape) == 1:
        actions = actions.reshape(-1, 1)
    
    return d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )


def main():
    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------
    base_dir = os.getcwd()
    generalization_datasets_path = os.path.join(base_dir, "results/data")
    dt_models_base_dir = os.path.join(base_dir, "results/models/DT")
    os.makedirs(generalization_datasets_path, exist_ok=True)
    
    env_5 = k2d(n=5, render_mode="human")
    env_7 = k2d(n=7, render_mode="human")
    env_10 = k2d(n=10, render_mode="human")
    
    # ---------------------------------------------------------
    # Generate Generalization Datasets
    # ---------------------------------------------------------
    print("\n=== Generate Generalization Datasets ===")
    
    # Ground Truth: Optimal policy, only winning trajectories (NORMALIZED)
    gt_5_path = os.path.join(generalization_datasets_path, "ground_truth_5_norm.h5")
    gt_7_path = os.path.join(generalization_datasets_path, "ground_truth_7_norm.h5")
    gt_5_7_path = os.path.join(generalization_datasets_path, "ground_truth_5_7_norm.h5")
    
    if os.path.exists(gt_5_path):
        print(f"Ground truth 5 (normalized) already exists: {gt_5_path}")
    else:
        print("Generating ground truth 5 (normalized)...")
        gt_5_trajs = generate_dataset(env_5, num_episodes=1000, policy='optimal', filter_wins=True)
        gt_5_dataset = convert_to_mdp_dataset_normalized(gt_5_trajs, grid_size=5)
        gt_5_dataset.dump(gt_5_path)
        print(f"Saved {len(gt_5_trajs)} normalized trajectories to {gt_5_path}")
   
    if os.path.exists(gt_7_path):
        print(f"Ground truth 7 (normalized) already exists: {gt_7_path}")
    else:
        print("Generating ground truth 7 (normalized)...")
        gt_7_trajs = generate_dataset(env_7, num_episodes=1000, policy='optimal', filter_wins=True)
        gt_7_dataset = convert_to_mdp_dataset_normalized(gt_7_trajs, grid_size=7)
        gt_7_dataset.dump(gt_7_path)
        print(f"Saved {len(gt_7_trajs)} normalized trajectories to {gt_7_path}")
    
    # Concatenate ground truth 5 and ground truth 7 (both already normalized)
    if os.path.exists(gt_5_7_path):
        print(f"Combined dataset (normalized) already exists: {gt_5_7_path}")
    else:
        print("Concatenating normalized ground_truth_5 and ground_truth_7...")
        gt_combined = concatenate_datasets([gt_5_path, gt_7_path])
        gt_combined.dump(gt_5_7_path)
        print(f"Saved combined normalized dataset to {gt_5_7_path}")
    
    # ---------------------------------------------------------
    # Train Decision Transformers
    # ---------------------------------------------------------
    print("\n=== Decision Transformers Training ===")
    # Train on: individual sizes (5, 7) and combined (5+7), then test on size 10
    generalization_datasets = ["ground_truth_5_norm", "ground_truth_5_7_norm"]
    dt_models_list = []

    for dataset_name in generalization_datasets:
        dt_save_path = os.path.join(dt_models_base_dir, f"generalization/dt_model_{dataset_name}.d3")
        os.makedirs(os.path.dirname(dt_save_path), exist_ok=True)
        
        dt_models_list.append([f"dt_{dataset_name}", dt_save_path])
        
        if os.path.exists(dt_save_path):
            print(f"DT Model already exists: {dt_save_path}")
            continue
        
        dataset_path = os.path.join(generalization_datasets_path, f"{dataset_name}.h5")
        print(f"\n--- Training DT on {dataset_name} ---")
        
        dt_config = {
            "environment": {
                "grid_size": 10
            },
            "pipeline": {
                "steps": ["train_dt"]
            },
            "dt": {
                "dataset_path": dataset_path,
                "context_size": 10,
                "n_steps": 1000,
                "n_steps_per_epoch": 100,
                "batch_size": 64,
                "learning_rate": 1e-4,
                "save_path": dt_save_path,
                "log_dir": f"logs/generalization/dt_{dataset_name}"
            }
        }
        
        train_dt(dt_config)
        print(f"DT Model saved to: {dt_save_path}")

    # ---------------------------------------------------------
    # Evaluation (using normalized observations)
    # ---------------------------------------------------------
    print("\n=== Evaluation (testing on size 10 - generalization test) ===")
    eval_config_path = os.path.join(base_dir, "src/experiments/generalization/configs/evaluate_generalization.json")
    
    eval_config = {
        "grid_size": 10,
        "target_return": 10.0,
        "models": dt_models_list
    }
    save_json(eval_config_path, eval_config)
    
    for i in range(10):
        for model_name, model_path in dt_models_list:
            print(f"\nEvaluating {model_name} on size 10 (generalization)...")
            model = load_dt(model_path)
            evaluate_dt_normalized(env_10, model, grid_size=10, target_return=10.0)

        for model_name, model_path in dt_models_list:
            print(f"\nEvaluating {model_name} on size 7 (in-distribution)...")
            model = load_dt(model_path)
            evaluate_dt_normalized(env_7, model, grid_size=7, target_return=10.0)
    
    print("\nAll evaluations completed successfully!")

if __name__ == "__main__":
    main()
