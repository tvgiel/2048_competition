from game2048 import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class DQN_2048(nn.Module):
    def __init__(self):
        super(DQN_2048, self).__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten for the fully connected layer
        x = torch.relu(self.fc1(x))
        return self.fc2(x) # Returns the predicted value of the 4 actions
    
    
class Agent:
    def __init__(self,):
        self.model = DQN_2048()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        # Memory stores past experiences: (state, action, reward, next_state, done)
        self.memory = deque(maxlen=10000)
        
        # Epsilon controls exploration. 
        # 1.0 means 100% random moves. It decays over time as the AI gets smarter.
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.gamma = 0.9 # Discount factor for future rewards

    def preprocess_state(self, state):
        """Flattens the 4x4 board and applies log2 scaling."""
        state = np.array(state)
        safe_state = np.where(state == 0, 1, state)
        log_state = np.log2(safe_state).astype(int)
        
        # one-hot encoding for up to 16 channels
        one_hot = np.zeros((16, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                val = log_state[i, j]
                if val < 16:
                    one_hot[val, i, j] = 1.0
                    
        return torch.FloatTensor(one_hot)

    def act(self, state, valid_actions):
        """Chooses an action based on the current board."""
        if random.random() < self.epsilon:
            # Explore: pick a random valid move
            return random.choice(valid_actions) if valid_actions else 0
        else:
            # Exploit: ask the Neural Network for the best move
            state_tensor = self.preprocess_state(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                
            # Pick the action with the highest Q-value that is actually a valid move
            # We set invalid moves to a massive negative number so they aren't picked
            q_values = q_values.numpy()[0]
            for action in range(4):
                if action not in valid_actions:
                    q_values[action] = -float('inf')
                    
            return np.argmax(q_values)

    def train_step(self, batch_size=64):
        """Trains the neural network on a random batch of past experiences."""
        if len(self.memory) < batch_size:
            return # Not enough memories to train yet

        # Grab a random batch of memories
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_tensor = self.preprocess_state(state).unsqueeze(0)
            next_state_tensor = self.preprocess_state(next_state).unsqueeze(0)
            
            # What did the network predict the score would be?
            current_q = self.model(state_tensor)[0, action]
            
            # What was the actual score (reward + predicted future reward)?
            if done:
                target_q = torch.tensor(float(reward))
            else:
                with torch.no_grad():
                    max_future_q = torch.max(self.model(next_state_tensor))
                    target_q = reward + (self.gamma * max_future_q)

            # Calculate how wrong the network was (the loss) and update the weights
            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Slowly reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            
def instantiate_and_train():
    env = Game2048Env()
    agent = Agent()

    episodes = 10
    save_every = 10

    # Load the model if it exists
    for model_nr in range(500,0,-save_every):
        model_path = f"model_checkpoint_episode_{model_nr}.pt"
        if os.path.exists(model_path):
            agent.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        break
    else:
        print("No saved model found. Starting with a new model.")

    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            valid_actions = env.get_available_actions()
            
            # 1. AI chooses an action
            action = agent.act(state, valid_actions)
            
            # 2. Environment takes a step
            next_state, reward, done, info = env.step(action)
            
            # 3. Save the result to memory
            agent.memory.append((state, action, reward, next_state, done))
            
            # 4. Train the brain
            agent.train_step()
            
            state = next_state
            
        # print(f"Episode {episode + 1}/{episodes} - Score: {info['score']} - Max Tile: {info['highest_tile']} - Epsilon: {agent.epsilon:.2f}")
        
        # Save model every 10 episodes
        if (episode + 1) % 10 == 0:
            torch.save(agent.model.state_dict(), f"model_checkpoint_episode_{episode + 1}.pt")
            print(f"Model saved at episode {episode + 1}")
        
        return agent
    
def play_game(agent):
    env = Game2048Env()
    state = env.reset()

    while not env.done:
        valid_actions = env.get_available_actions()
        action = agent.act(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        state = next_state
    return info['score'], info['highest_tile']
    

    