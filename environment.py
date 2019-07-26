#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:26:31 2017

@author: farismismar
"""

import math
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

# An attempt to follow
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

class SON_environment:
    
    def __init__(self, seed):
        self.num_states = 3
        self.num_actions = 6
        
        self.action_space = spaces.Discrete(self.num_actions) # action size is here
        self.observation_space = spaces.Discrete(self.num_states) # action size is here
        
        self.seed(seed=seed)

        self.step_count = 0 # which step
        
        self.state = 0
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        self.state = 0
        self.step_count = 0
        return np.array(self.state)

    def close(self):
        self.reset()
    
    def step(self, action):
        
        # Check if action is integer
      #  if isinstance(action, np.ndarray):
       #     action = action[0]
        
        # Update the state based on the action
        # This is handled by Matlab's function.

        self.step_count += 1
        return self.state, None, None, None