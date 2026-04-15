import time
from game2048 import Game2048Env
import random

class RandomAI:
    def __init__(self):
        # Priority order: random
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.priorities = [3, 0, 1, 2]

    def get_best_move(self, env):
        available_actions = env.get_available_actions()
        
        return random.choice(available_actions)
                
        return -1 # No valid moves available

def initiate_and_train():
    return


def play_game(agent):
    env = Game2048Env()
    env.reset()
    
    ai = RandomAI()
    
    action_map = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    move_count = 0
    
    
    while not env.done:
        best_action = ai.get_best_move(env)
        
        if best_action == -1:
            break
            
        _, reward, done, info = env.step(best_action)
        move_count += 1
        
        # # Print state occasionally or at the end
        # if move_count % 20 == 0 or done:
        #     print(f"Move {move_count}: played {action_map[best_action]}")
        #     env.render()
            
    return info['score'], info['highest_tile']


