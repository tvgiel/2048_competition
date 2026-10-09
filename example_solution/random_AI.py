import time
from game2048 import Game2048Env
import random

class RandomAI:
    def __init__(self):
        # Priority order: random
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.priorities = [3, 0, 1, 2]

    def act(self, game_state, valid_actions):
        return random.choice(valid_actions)

    def get_best_move(self, env):
        return self.act(env.get_state(), env.get_available_actions())

def train_and_instantiate():
    return RandomAI()




def play_game(agent):
    env = Game2048Env()
    state = env.reset()

    while not env.done:
        valid_actions = env.get_available_actions()
        action = agent.act(state, valid_actions)
        state, reward, done, info = env.step(action)

    return info['score'], info['highest_tile']


