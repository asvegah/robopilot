'''
file: gym_real.py
author: Tawn Kramer
date: 2019-01-24
desc: Control a real robopilot robot via the gym interface
'''
import os
import time

import gym
import numpy as np
from gym import error, spaces, utils

from .remote_controller import RobopilotRemoteContoller


class RobopilotRealEnv(gym.Env):
    """
    OpenAI Gym Environment for a real Robopilot
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION_NAMES = ["steer", "throttle"]
    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0
    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 5.0
    VAL_PER_PIXEL = 255

    def __init__(self, time_step=0.05, frame_skip=2):

        print("starting RobopilotGym env")
        
        try:
            robopilot_name = str(os.environ['ROBOPILOT_NAME'])
        except:
            robopilot_name = 'my_robot1234'
            print("No ROBOPILOT_NAME environment var. Using default:", robopilot_name)

        try:
            mqtt_broker = str(os.environ['ROBOPILOT_MQTT_BROKER'])
        except:
            mqtt_broker = "iot.eclipse.org"
            print("No ROBOPILOT_MQTT_BROKER environment var. Using default:", mqtt_broker)
            
        # start controller
        self.controller = RobopilotRemoteContoller(robopilot_name=robopilot_name, mqtt_broker=mqtt_broker)
        
        # steering and throttle
        self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
            high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]), dtype=np.float32 )

        # camera sensor data
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.controller.get_sensor_size(), dtype=np.uint8)

        # Frame Skipping
        self.frame_skip = frame_skip

        # wait until loaded
        self.controller.wait_until_connected()
        

    def close(self):
        self.controller.quit()        

    def step(self, action):
        for i in range(self.frame_skip):
            self.controller.take_action(action)
            time.sleep(0.05)
            observation = self.controller.observe()
            reward, done, info = 0.1, False, None
        return observation, reward, done, info

    def reset(self):
        observation = self.controller.observe()
        reward, done, info = 0.1, False, None
        return observation

    def render(self, mode="human", close=False):
        if close:
            self.controller.quit()

        return self.controller.observe()

    def is_game_over(self):
        return False
