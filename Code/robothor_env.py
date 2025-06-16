import os
import glob
import gym
from gym import spaces
import numpy as np
import ai2thor.controller
import cv2  # For resizing frame

class RoboThorEnv(gym.Env):
    def __init__(self, scene="FloorPlan_Train1_1"):
        super(RoboThorEnv, self).__init__()

        # Initialize action log
        self.logs = []

        # Cleanup old pipes:
        for f in glob.glob("/tmp/thor_*"):
            try:
                os.remove(f)
            except:
                pass

        # Start controller (start() is automatic now)
        self.controller = ai2thor.controller.Controller()

        # Reset scene using LoCoBot agent mode (to avoid warning)
        self.controller.reset(scene, agentMode='locobot')

        # Define action space
        self.actions = ["MoveAhead", "RotateRight", "RotateLeft", "LookUp", "LookDown"]
        self.action_space = spaces.Discrete(len(self.actions))

        # Define observation space (resized to 84x84 RGB)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def step(self, action):
        action_name = self.actions[action]
        event = self.controller.step(action=action_name)

        done = not event.metadata['lastActionSuccess']
        reward = 1.0 if not done else -1.0

        obs = self._get_observation(event)

        # Log each step
        self.logs.append({
            "action": action_name,
            "success": event.metadata['lastActionSuccess'],
            "position": event.metadata['agent']['position'],
            "rotation": event.metadata['agent']['rotation'],
        })

        return obs, reward, done, {}

    def reset(self):
        # Reset scene and clear logs
        self.logs = []
        self.controller.reset("FloorPlan_Train1_1", agentMode='locobot')
        event = self.controller.step(action="Pass")
        obs = self._get_observation(event)
        return obs

    def _get_observation(self, event):
        frame = event.frame
        frame = cv2.resize(frame, (84, 84))
        return frame

    def render(self, mode='human'):
        pass

    def close(self):
        self.controller.stop()
