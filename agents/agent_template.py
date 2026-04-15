from game2048 import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class AI_model():
    def act(game_state, valid_actions):
        pass
    pass

# some models don't have to be trained. If that's the case, just return the untrained model.
def instantiate_and_train():
    agent = AI_model()
    train_agent(agent, episodes=500)
    return agent


    
def play_game(agent):
    # always load the game environment first so then the agent can interact with it
    env = Game2048Env()
    state = env.reset()

    # Do not change the code below, as this is the main loop of the game. The agent should interact with the environment through this loop.
    while not env.done:
        valid_actions = env.get_available_actions()
        action = agent.act(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        state = next_state
    return info['score'], info['highest_tile']

    