 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:20:05 2017

@author: farismismar
"""

import random
import numpy as np
from collections import deque

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tick

memory = deque(maxlen=2000)

from environment import SON_environment
#from QLearningAgent import QLearningAgent as QLearner
from DQNLearningAgent import DQNLearningAgent as QLearner

seed = 0 # change in Top_File.m also

random.seed(seed)
np.random.seed(seed)

batch_size = 12
state_count = 3
action_count_b = 6

env = None
agent = None 

# This is the entry point to the simulation
def env_reset_wrapper():
    random.seed(seed)
    np.random.seed(seed)
    global env
    global agent 
    state = env.reset()
    return state

def agent_get_exploration_rate_wrapper():
    global env
    global agent 
    return agent.exploration_rate;

def set_environment(state_size, action_size):
    state_count = int(state_size)
    action_count_b = int(action_size)
    global env
    global agent 
    
    env = SON_environment(seed=seed)
    agent = QLearner(seed=seed)

def env_step_wrapper(action):
    global env
    global agent 
    return env.step(action)

def agent_act_wrapper(state):
    global env
    global agent
    state = np.asarray(state)
    return agent.act(state)

def agent_begin_episode_wrapper(state):
    global env
    global agent 
    state = np.asarray(state)
    return agent.begin_episode(state)

def agent_replay_wrapper():
    global env
    global agent 
    global batch_size
    global memory
    [q, loss] = agent.replay(memory, batch_size)

    return [q, loss]
 
def agent_memory_length_diff_wrapper():
    global env
    global agent 
    global batch_size
    global memory
    return (len(memory) - batch_size)

def agent_remember_wrapper(state, action, reward, next_state, done):
    global env
    global agent 
    global memory
    state = np.asarray(state)
    next_state = np.asarray(next_state)
    action = np.asarray(action)
    state = int(state[0])
    next_state = int(next_state[0])
    action = int(action[0])
    reward = int(reward)
    done = int(done)
    m = np.array([state, action, reward, next_state, done])

    #f=open('/Users/farismismar/Desktop/memory.csv','ab')
    #np.savetxt(f, m.T, fmt="%s", delimiter=",")
    #f.close()

    memory = agent.remember(memory, state, action, reward, next_state, done)
    # Checking if the memory populated properly... yes.
#    m = np.array(memory)
#f=open('/Users/farismismar/Desktop/memory.csv','ab')
 #   np.savetxt(f, m, fmt="%s", delimiter=",")
#    f.close()
