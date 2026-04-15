import numpy as np
import random
from game2048 import *
from pathlib import Path

import numpy as np
import random
import pickle

class AfterstateRAMNetAgent:
    def __init__(self, alpha=0.05, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0.1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        
        # Track 4 rows and 4 columns
        self.tuples = [
            [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], 
            [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]  
        ]
        
        self.ram_nodes = [{} for _ in self.tuples]

    def _get_addresses(self, afterstate):
        """Generates addresses based ONLY on the board state, no actions attached."""
        flat_state = np.where(afterstate.flatten() == 0, 1, afterstate.flatten())
        log_state = np.log2(flat_state).astype(int)
        
        # The address is purely the tuple of tile values
        return [tuple(log_state[i] for i in t) for t in self.tuples]

    def get_value(self, afterstate):
        """Calculates the value of a specific board layout."""
        addresses = self._get_addresses(afterstate)
        return sum(self.ram_nodes[i].get(addr, 0.0) for i, addr in enumerate(addresses))

    def simulate_afterstate(self, state, action):
        """
        The Forward Model: Simulates sliding the board without spawning a new tile.
        This allows the AI to "imagine" the consequences of its moves.
        """
        # We reuse your environment's clever rotation trick!
        test_board = np.rot90(state.copy(), k=(action + 1) % 4)
        reward = 0
        
        for i in range(4):
            # 1. Slide and merge
            non_zero = [x for x in test_board[i] if x != 0]
            merged_row = []
            skip = False
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if j < len(non_zero) - 1 and non_zero[j] == non_zero[j+1]:
                    merged_val = non_zero[j] * 2
                    merged_row.append(merged_val)
                    reward += merged_val
                    skip = True
                else:
                    merged_row.append(non_zero[j])
            
            merged_row.extend([0] * (4 - len(merged_row)))
            test_board[i] = np.array(merged_row)
            
        test_board = np.rot90(test_board, k=-((action + 1) % 4))
        return test_board, reward

    def act(self, state, valid_actions):
        """Chooses the action that leads to the highest-valued Afterstate."""
        if not valid_actions: 
            return 0
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
            
        best_action = valid_actions[0]
        max_value = -float('inf')
        
        # Evaluate all legal moves by looking at their Afterstates
        for action in valid_actions:
            afterstate, immediate_reward = self.simulate_afterstate(state, action)
            
            # Total Value = The points scored during the slide + The estimated value of the resulting board
            value = immediate_reward + self.get_value(afterstate)
            
            if value > max_value:
                max_value = value
                best_action = action
                
        return best_action

    def update(self, current_afterstate, reward, next_afterstate):
        """Updates the RAM nodes using TD-Learning strictly between Afterstates."""
        current_value = self.get_value(current_afterstate)
        
        if next_afterstate is None:
            target_value = reward
        else:
            target_value = reward + self.gamma * self.get_value(next_afterstate)
            
        td_error = target_value - current_value
        update_step = (self.alpha * td_error) / len(self.tuples)
        
        addresses = self._get_addresses(current_afterstate)
        for i, addr in enumerate(addresses):
            if addr not in self.ram_nodes[i]:
                self.ram_nodes[i][addr] = 0.0
            self.ram_nodes[i][addr] += update_step
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            

def train():             
    env = Game2048Env()
    agent = AfterstateRAMNetAgent()

    episodes = 50000

    for episode in range(episodes):
        state = env.reset()
        done = False
        
        # We must determine the first afterstate before entering the main loop
        valid_actions = env.get_available_actions()
        action = agent.act(state, valid_actions)
        current_afterstate, immediate_reward = agent.simulate_afterstate(state, action)
        
        while not done:
            # Step the environment forward based on our chosen action
            next_state, _, done, info = env.step(action)
            valid_next_actions = env.get_available_actions()
            
            if done:
                # Game over, there is no next afterstate
                agent.update(current_afterstate, immediate_reward, None)
            else:
                # Plan the NEXT move to find the NEXT afterstate
                next_action = agent.act(next_state, valid_next_actions)
                next_afterstate, next_immediate_reward = agent.simulate_afterstate(next_state, next_action)
                
                # Learn the transition!
                agent.update(current_afterstate, immediate_reward, next_afterstate)
                
                # Shift variables forward for the next loop iteration
                action = next_action
                current_afterstate = next_afterstate
                immediate_reward = next_immediate_reward
                
        if episode % 100 == 0:
            print(f"Episode {episode} - Score: {info['score']} - Max Tile: {info['highest_tile']}")
            curr_file_name = Path(__file__).stem
            with open('agent_' + curr_file_name + '.pkl', 'wb') as f:
                pickle.dump(agent, f)
                

def play_game():
    print("Training finished! Playing a test game now...")
    # Disable exploration completely for playing
    agent.epsilon = 0.0
    
    
    env = Game2048Env()
    state = env.reset()
    env.render()

    while not env.done:
        valid_actions = env.get_available_actions()
        if not valid_actions:
            break
            
        action = agent.act(state, valid_actions)
        state, reward, done, info = env.step(action)
        
        # Render the board to watch the AI play
        env.render()
    return env.info