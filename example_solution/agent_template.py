from game2048 import *

import random

class AI_model():
    def act(self, game_state, valid_actions):
        return the_choice_your_agent_makes

# some models don't have to be trained. If that's the case, just return the untrained model.
def train_and_instantiate():
    return AI_model()


    
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


# this will play one game of 2048
def main():
    agent = train_and_instantiate()
    play_game(agent)
    