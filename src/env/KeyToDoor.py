import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EnvSpec:
    """Simple spec class for environment metadata."""
    def __init__(self, max_episode_steps):
        self.max_episode_steps = max_episode_steps


class KeyToDoorEnv(gym.Env):
    """
    3-room grid-based Key-to-Door environment.

    Rooms:
      - Room 0: key at (self.grid_size-1,self.grid_size/2)
      - Room 1: empty
      - Room 2: door at (0,self.grid_size/2)

    Rules:
      - Each room is a nxn grid (coords 0..n-1) 
      - Agent always spawns in each room at (n-1,0)
      - Agent has 2n steps per room, then teleports to next room
      - Success only if agent reaches door (room 2, at (0,self.grid_size/2)) *while holding key*
      - Small negative reward per step (-0.01)
      - Terminal reward +10 on success

    Actions:
      0 = up
      1 = down
      2 = left
      3 = right
      4 = pick
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, n=4):
        assert n > 0
        self.render_mode = render_mode
        self.max_room_steps = 2 * n
        self.observation_space = spaces.Dict({
            "room": spaces.Discrete(3),
            "pos": spaces.Box(low=0, high=n-1, shape=(2,), dtype=np.int32),
            "has_key": spaces.Discrete(2),
        })

        self.action_space = spaces.Discrete(5)

        self.grid_size = n
        self.mid = (self.grid_size - 1) // 2

        # Create spec with max_episode_steps (3 rooms * max_room_steps)
        # Maximum episode steps = 3 rooms * steps per room
        max_episode_steps = 3 * self.max_room_steps
        self.spec = EnvSpec(max_episode_steps)
        self.room = None
        self.pos = None
        self.has_key = None
        self.room_steps = None

    def _get_obs(self):
        # Clamp room to valid range [0, 2] to prevent out-of-bounds errors
        room = min(2, max(0, self.room))
        return {
            "room": np.int32(room),
            "pos": self.pos.copy(),
            "has_key": np.int32(self.has_key),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.room = 0
        self.pos = np.array([self.grid_size - 1, 0], dtype=np.int32)
        self.has_key = 0
        self.room_steps = 0
        return self._get_obs(), {}

    def step(self, action):
        reward = -0.01
        terminated = False
        truncated = False

        # Movement inside grid
        if action == 0:      # up
            self.pos[0] = max(0, self.pos[0] - 1)
        elif action == 1:    # down
            self.pos[0] = min(self.grid_size-1, self.pos[0] + 1)
        elif action == 2:    # left
            self.pos[1] = max(0, self.pos[1] - 1)
        elif action == 3:    # right
            self.pos[1] = min(self.grid_size-1, self.pos[1] + 1)
        elif action == 4:    # pick key
            if self.room == 0 and (self.pos[0] == self.grid_size-1 and self.pos[1] == self.mid):
                self.has_key = 1

        # Success check (only room 2)
        if self.room == 2 and self.has_key == 1 and (self.pos[0] == 0 and self.pos[1] == self.mid):
            reward = 10
            terminated = True

        # Handle room step counter
        self.room_steps += 1
        if not terminated and self.room_steps >= self.max_room_steps:
            self.room += 1
            if self.room >= 3:
                truncated = True
            else:
                self.pos = np.array([self.grid_size-1,0], dtype=np.int32)
                self.room_steps = 0

        return self._get_obs(), reward, terminated, truncated, {}

    @staticmethod
    def print_action(action):
        action_str = ""
        if action == 0:      # up
            action_str = "up"
        elif action == 1:    # down
            action_str = "down"
        elif action == 2:    # left
            action_str = "left"
        elif action == 3:    # right
            action_str = "right"
        elif action == 4:    # pick key
            action_str = "pick up key"
        print("agent's action is: ", action_str)
        
    def render(self):
        if self.render_mode != "human" or self.room == 3:
            return

        # Create empty grid
        grid = [[" . " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Add key
        if self.room == 0 and self.has_key == 0:
            grid[self.grid_size - 1][self.mid] = " K "

        # Add door
        if self.room == 2:
            grid[0][self.mid] = " D "

        # Add agent
        ax, ay = self.pos
        grid[ax][ay] = " A "

        # Dynamic horizontal separator
        row_border = "+" + "+".join(["---"] * self.grid_size) + "+"

        # Print
        print(f"Room {self.room} | has_key={self.has_key}")
        print(row_border)
        for row in grid:
            print("|" + "|".join(row) + "|")
            print(row_border)


    def close(self):
        pass
