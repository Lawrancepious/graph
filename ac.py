import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


class BinPackingEnv(gym.Env):
    """
    Custom Environment for the bin-packing problem.
    """

    def __init__(self):
        super(BinPackingEnv, self).__init__()

        # Define bin dimensions
        self.bin_width = 80
        self.bin_height = 40

        # Define rectangles (width, height)
        self.rectangles = [
            (10, 5),
            (15, 10),
            (20, 10),
            (10, 15),
            (5, 10),
            (10, 10),
            (5, 5),
            (15, 15),
            (10, 20),
        ]

        # Action space: [rectangle_index, x_coordinate, y_coordinate]
        self.action_space = spaces.MultiDiscrete(
            [len(self.rectangles), self.bin_width, self.bin_height]
        )

        # Observation space: Current state of the bin (binary grid representation)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.bin_height, self.bin_width), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.bin = np.zeros((self.bin_height, self.bin_width), dtype=np.float32)
        self.placed = []  # Track placed rectangles
        return self._get_observation()

    def _get_observation(self):
        """
        Returns the current state of the bin as the observation.
        """
        return self.bin.copy()

    def step(self, action):
        """
        Execute an action and return the new state, reward, done flag, and info.
        """
        rect_idx, x, y = action
        rect = self.rectangles[rect_idx]
        w, h = rect

        if self._is_valid_placement(x, y, w, h):
            # Place the rectangle
            self.bin[y : y + h, x : x + w] = 1
            self.placed.append((rect_idx, x, y, w, h))
            reward = w * h  # Reward based on area successfully placed
        else:
            reward = -1  # Penalize invalid action

        done = len(self.placed) == len(
            self.rectangles
        )  # Episode ends when all rectangles are placed
        return self._get_observation(), reward, done, {}

    def _is_valid_placement(self, x, y, w, h):
        """
        Checks if the rectangle can be placed at (x, y) without overlap.
        """
        if x + w > self.bin_width or y + h > self.bin_height:
            return False
        if np.any(self.bin[y : y + h, x : x + w] == 1):
            return False
        return True

    def render(self, mode="human"):
        """
        Visualize the current state of the bin and placements.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(self.bin, cmap="gray", origin="upper")
        for _, x, y, w, h in self.placed:
            rect = plt.Rectangle((x, y), w, h, edgecolor="blue", facecolor="none", lw=2)
            ax.add_patch(rect)
        plt.show()


# Initialize the environment
env = BinPackingEnv()

# Train the model using PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("bin_packing_ppo")

# Test the trained model
model = PPO.load("bin_packing_ppo")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)  # Use the trained model to predict actions
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Reward: {reward}")
