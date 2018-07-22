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

'''
def run_agent(env, X, plotting = False):
    global env
    global agent 
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = TTI_count

    succ = [] # a list to save the good ones
    successful = False
    for episode_index in np.arange(max_episodes_to_run):
        state = env.reset()
        reward = R_min
        action = agent.begin_episode(state)
    
        cell_score = baseline_SINR_dB

        # Recording arrays
        state_progress = ['start', cell_score]
        action_progress = ['start', 0]
        score_progress = [cell_score]
        
        for timestep_index in range(max_timesteps_per_episode):
            # Let Network (Player A) function totally random
            cell_score += get_A_contrib() #player_A_contribs[np.random.randint(action_count_a)]
            #state_progress.append(np.round(cell_score, 1))
            #score_progress.append(np.round(cell_score, 1))
            action_progress.append('network')  # The network action is empty
            
            # Perform the power control action and observe the new state.
            action = agent.act(state, reward)                       
            next_state, reward, _, _ = env.step(action)
                       
            power_command = player_B_contribs[next_state[0]]

            pt_current *= 10 ** (power_command / 10.) # the current ptransmit in mW due to PC
            
            # ptransmit cannot exceed pt, max.
            if (pt_current >= pt_max):
                pt_current = pt_max
            else:
                cell_score += power_command            
       
            aborted = (cell_score < SINR_MIN)
            done = (cell_score >= final_SINR_dB)
            
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state
        
            # Rewards are here
            if done:
                if timestep_index < 15: # premature ending -- cannot finish sooner than 15 episodes
                    aborted = True
                else:                       # ending within time.
                   successful = True
                   reward = R_max
                   aborted = False
            
            #print(reward)
#            
            # I truly care about the net change: network - PC
            action_progress.append(state[0])
            score_progress.append(np.round(cell_score, 2))
            state_progress.append(np.round(cell_score, 2))            
                                
            if aborted == True:
                reward = R_min
                action_progress.append('ABORTED')
                state_progress.append('ABORTED')
                
            if (done or aborted):
                print("Episode {0} finished after {1} timesteps (and epsilon = {2:0.3}).".format(episode_index + 1, timestep_index + 1, agent.exploration_rate))
                action_progress.append('end')
                state_progress.append('end')
                
                print('Action progress: ')
                print(action_progress)
                print('SINR progress: ')
                print(state_progress) # this is actually the SINR progress due to the score
                
                 # Do some nice plotting here
                fig = plt.figure()
#                sinr_min = np.min(score_progress)
#                sinr_max = np.max(score_progress)

                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                plt.xlabel('Transmit Time Interval (1 ms)')

                # Only integers                                
                ax = fig.gca()
                ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
                ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode + 1))

                ax.set_autoscaley_on(False)

                plt.plot(score_progress, marker='o', linestyle='--', color='b')
                plt.xlim(xmin=0, xmax=max_timesteps_per_episode)

                plt.axhline(y=SINR_MIN, xmin=0, color="red", linewidth=1.5)
                plt.axhline(y=final_SINR_dB, xmin=0, color="green",  linewidth=1.5)
                plt.ylabel('Average DL Received SINR (dB)')
                plt.title('Episode {0} / {1} ($\epsilon = {2:0.3}$)'.format(episode_index + 1, max_episodes_to_run, agent.exploration_rate))
                plt.grid(True)
                plt.ylim(-8,10)
                plt.savefig('figures/episode_{}.pdf'.format(episode_index + 1), format="pdf")
                if (plotting):
                    plt.show(block=True)
                plt.close(fig)
                
                if (plotting):
                    plot_actions(action_progress, episode_index + 1)
                    
                print('-'*80)       
                break                    

        if (successful):
            succ.append(episode_index+1)
        
        # For multi-plotting purposes
        if (episode_index + 1 == 725 or episode_index + 1 == 2):
            file = open("plot_sinr.txt","a") 
            for item in score_progress:
                file.write("{},".format(item))
            file.write("\n")
            file.close()

        # Remove these four lines after finding the correct episode            
        if (successful and episode_index + 1 >= 2000):
   #         break # This is the number I truly need to run my program for.
        #We found xxx episodes required.
            None
        else:
            successful = False

        # train the agent with the experience of the episode
        if len(agent.memory) > agent.batch_size:
            agent.replay(agent.batch_size)
            # Show the losses here
            losses = agent.get_losses()
            
    # Now plot the losses NOT for vanilla Q
    try:
        print('Successful episodes')
        print(succ)
        
        if len(losses) > 0:
            fig = plt.figure(figsize=(7,5))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.plot(losses, color='k')
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.title('Losses vs Episode')
            
            ax = fig.gca()
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
        
            plt.grid(True)
            plt.ylim(ymin=-400)
            plt.axhline(y=0, xmin=0, color="gray", linestyle='dashed', linewidth=1.5)
            plt.savefig('figures/loss_episode.pdf', format="pdf")
            if (plotting):
                plt.show(block=True)
            plt.close(fig)
    except:
        None # technically do nothing-- this is the vanilla Q
        
    if (not successful):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum.".format(max_episodes_to_run))
    
    #plt.ioff()
    
########################################################################################
    
#run_agent(env, True)

########################################################################################
'''