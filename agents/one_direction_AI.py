import time
from game2048 import Game2048Env

class SimplePriorityAI:
    def __init__(self):
        # Priority order: Left (3), Up (0), Right (1), Down (2)
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.priorities = [3, 0, 1, 2]

    def get_best_move(self, env):
        available_actions = env.get_available_actions()
        
        for action in self.priorities:
            if action in available_actions:
                return action
                
        return -1 # No valid moves available



def play_game(ai):
    env = Game2048Env()
    env.reset()
    
    
    action_map = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    move_count = 0
    
    
    while not env.done:
        best_action = ai.get_best_move(env)
        
        if best_action == -1:
            break
            
        _, reward, done, info = env.step(best_action)
        move_count += 1
        
    return info['score'], info['highest_tile']


