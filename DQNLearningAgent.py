#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:02:40 2019

@author: farismismar
"""

# Used from: https://keon.io/deep-q-learning/
# https://github.com/keon/deep-q-learning/blob/master/dqn.py
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/reinforcement_learning/deep_q_network.py

# Check some more here: https://github.com/leimao/OpenAI_Gym_AI/tree/master/CartPole-v0/Deep_Q-Learning
# This adds a means to compute AverageQ as a sign of experience.

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

# Deep Q-learning Agent
import random
import numpy as np

from keras.models import Sequential
from keras.callbacks import History 
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras import backend as K
import tensorflow as tf

class DQNLearningAgent:
    def __init__(self, seed,
                 discount_factor=0.995,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.999):
                               
        #####self.memory = deque(maxlen=2000)
        self.gamma = discount_factor    # discount rate
        self.exploration_rate = exploration_rate #/ exploration_decay_rate # exploration rate
        self.exploration_rate_min = 0.010
        self.exploration_rate_decay = exploration_decay_rate
        self.learning_rate = 0.01 # this is eta for SGD

        self._state_size = 3 # unchange
        self._action_size = 6 # unchange
        
        # Add a few lines to caputre the seed for reproducibility.
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        gpu_available = tf.test.is_gpu_available()
        if (gpu_available == False):
            print('WARNING: No GPU available.  Will continue with CPU.')

        self.model = self._build_model()
                
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_rate_decay
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
            
        # return an action at random
        action = random.randrange(self._action_size)

        return action

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # This is a state-to-largest Q converter to find best action, basically
        model = Sequential()
        model.add(Dense(24, input_dim=self._state_size, activation='sigmoid'))
        model.add(Dense(24, activation='sigmoid'))
        model.add(Dense(self._action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True))
        
        return model
    
    def _construct_training_set(self, replay):
        # Select states and next states from replay memory
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])

        states = np.tile(states, (self._state_size, 1)).T
        new_states = np.tile(new_states, (self._state_size, 1)).T

        #np.savetxt("/Users/farismismar/Desktop/s.csv", states, fmt="%s", delimiter=",")
        #np.savetxt("/Users/farismismar/Desktop/ns.csv", new_states, fmt="%s", delimiter=",")
        
        # Predict the expected Q of current state and new state using DQN
        with tf.device('/gpu:0'):
            Q = self.model.predict(states)
            Q_new = self.model.predict(new_states)

        replay_size = len(replay)
        X = np.empty((replay_size, self._state_size))
        y = np.empty((replay_size, self._action_size))
        
        # Construct training set
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]
            action_r = int(action_r) # I added this.
            target = Q[i]
            target[action_r] = reward_r
            # If we're done the utility is simply the reward of executing action a in
            # state s, otherwise we add the expected maximum future reward as well
            if not done_r:
                target[action_r] += self.gamma * np.amax(Q_new[i])

            X[i] = state_r
            y[i] = target

        return X, y
        
    def remember(self, memory, state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))
        # Make sure we restrict memory size to specified limit
        if len(memory) > 2000:
            memory.pop(0)
        
        return memory

    def act(self, state):
        # Exploration/exploitation: choose a random action or select the best one.
        if np.random.uniform(0, 1) <= self.exploration_rate:
            return random.randrange(self._action_size)
       
        state = np.tile(state, (self._state_size, 1)).T
        #states = states[:,0]
        with tf.device('/gpu:0'):
            act_values = self.model.predict(state)
            
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, memory, batch_size):
        minibatch = random.sample(memory, batch_size)
        
        X, y = self._construct_training_set(minibatch)
        with tf.device('/gpu:0'):
            loss = self.model.train_on_batch(X, y)
        
        _q = np.mean(y)
        return [_q, loss]
    
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        return

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)