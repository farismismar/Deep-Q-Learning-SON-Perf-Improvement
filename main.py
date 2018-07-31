#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:20:05 2017

@author: farismismar
"""

import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tick

from environment import SON_environment
#from QLearningAgent import QLearningAgent as QLearner
from DQNLearningAgent import DQNLearningAgent as QLearner

seed = 3 # change in Top_File.m also

random.seed(seed)
np.random.seed(seed)

batch_size = 32
state_count = 3
action_count_b = 5

env = SON_environment(random_state=seed, state_size=state_count, action_size=action_count_b)
agent = QLearner(seed=seed, state_size=state_count, action_size=action_count_b, batch_size=batch_size)

# This is the entry point to the simulation
def env_reset_wrapper():
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
    
    env = SON_environment(random_state=seed, state_size=state_count, action_size=action_count_b)
    agent = QLearner(seed=seed, state_size=state_count, action_size=action_count_b, batch_size=batch_size)

def env_step_wrapper(action):
    global env
    global agent 
    return env.step(action)

def agent_act_wrapper(state, reward):
    global env
    global agent 
    return agent.act(state, reward)

def agent_begin_episode_wrapper(state):
    global env
    global agent 
    return agent.begin_episode(state)

def agent_get_losses_wrapper():
    global env
    global agent 
    return agent.get_losses()

def agent_replay_wrapper():
    global env
    global agent 
    return agent.replay(agent.batch_size)
 
def agent_memory_length_diff_wrapper():
    global env
    global agent 
    return (len(agent.memory) - agent.batch_size)

def agent_remember_wrapper(state, action, reward, next_state, done):
    global env
    global agent 
    return agent.remember(state, action, reward, next_state, done)
