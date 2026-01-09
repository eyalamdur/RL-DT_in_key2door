from KeyToDoor import KeyToDoorEnv
import gymnasium as gym

def random_render(num_steps=30):
    # Create k2d environment with rendering enabled
    env = KeyToDoorEnv(render_mode="human")
    
    # Reset the environment
    obs, info = env.reset()
    print("Initial state:")
    print(f"Room: {obs['room']}, Position: {obs['pos']}, Has Key: {bool(obs['has_key'])}, Key Pos: {obs['key_pos']}")
    print()
    
    # Render initial state
    env.render()
    print()
    
    # Perform random moves
    total_reward = 0
    
    for step in range(num_steps):
        # Sample a random action (0-4: up, down, left, right, pick)
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Action={action} ({['up', 'down', 'left', 'right', 'pick'][action]})")
        print(f"Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        print(f"Room: {obs['room']}, Position: {obs['pos']}, Has Key: {bool(obs['has_key'])}, Key Pos: {obs['key_pos']}")
        
        # Render the environment
        env.render()
        print()
        
        # Check if episode is done
        if terminated:
            print("✓ SUCCESS! Agent reached the door with the key!")
            break
        elif truncated:
            print("✗ Episode truncated (ran out of steps)")
            break
    
    print("=" * 50)
    print(f"Random walk complete. Total reward: {total_reward:.2f}")
    env.close()


def winning_render(num_steps=30):
    # Create k2d environment with rendering enabled
    env = KeyToDoorEnv(render_mode="human")
    
    # Reset the environment
    obs, info = env.reset()
    print("Winning sequence - Initial state:")
    print(f"Room: {obs['room']}, Position: {obs['pos']}, Has Key: {bool(obs['has_key'])}, Key Pos: {obs['key_pos']}")
    print()
    
    # Render initial state
    env.render()
    print()
    
    # Predefined winning sequence:
    # Room 0: Start at (3,0), need to get to key at (2,2) and pick it at step 4
    # Room 1: Wait (empty room)
    # Room 2: Move to door at (0,2)
    
    # Sequence of actions to win:
    # Steps 1-3: Move from (3,0) to (2,2) in room 0
    # Step 4: Pick the key
    # Steps 5-10: Wait in room 0 (will transition to room 1 after step 10)
    # Steps 11-20: Wait in room 1 (will transition to room 2 after step 20)
    # Steps 21-25: Move from (3,0) to (0,2) in room 2 to reach door
    
    actions = [
        # Room 0: Move to key at (2,2)
        0,   # Step 1: up (3,0) -> (2,0)
        3,   # Step 2: right (2,0) -> (2,1)
        3,   # Step 3: right (2,1) -> (2,2)
        4,   # Step 4: pick key at (2,2)
        # Steps 5-10: Wait in room 0 (no-ops)
        0, 0, 0, 0, 0, 0,
        # Steps 11-20: Wait in room 1 (no-ops)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Room 2: Move to door at (0,2)
        0,   # Step 21: up (3,0) -> (2,0)
        0,   # Step 22: up (2,0) -> (1,0)
        0,   # Step 23: up (1,0) -> (0,0)
        3,   # Step 24: right (0,0) -> (0,1)
        3,   # Step 25: right (0,1) -> (0,2) - SUCCESS!
    ]
    
    total_reward = 0
    
    for step, action in enumerate(actions, 1):
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: Action={action} ({['up', 'down', 'left', 'right', 'pick'][action]})")
        print(f"Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        print(f"Room: {obs['room']}, Position: {obs['pos']}, Has Key: {bool(obs['has_key'])}")
        
        # Render the environment
        env.render()
        print()
        
        # Check if episode is done
        if terminated:
            print("✓ SUCCESS! Agent reached the door with the key!")
            break
        elif truncated:
            print("✗ Episode truncated (ran out of steps)")
            break
    
    print("=" * 50)
    print(f"Winning sequence complete. Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    print("=" * 50)
    print("Running random render...")
    print("=" * 50)
    random_render()
    
    print("\n")
    print("=" * 50)
    print("Running winning render...")
    print("=" * 50)
    winning_render()

