from game2048 import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_2048(nn.Module):
    def __init__(self):
        super(CNN_2048, self).__init__()
        
        # 1st Convolutional Layer
        # Input: 1 channel (the board), Output: 64 feature maps
        # Kernel: 2x2. A 4x4 board becomes 3x3 after a 2x2 convolution without padding.
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=2)
        
        # 2nd Convolutional Layer
        # Input: 64 channels, Output: 128 feature maps
        # Kernel: 2x2. The 3x3 maps become 2x2.
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        
        # Fully Connected (Dense) Layers
        # The output of conv2 is 128 channels of 2x2 grids. 
        # Flattened, that is 128 * 2 * 2 = 512 neurons.
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 4) # 4 outputs for Up, Right, Down, Left

    def forward(self, x):
        # x starts as shape: (Batch Size, 1, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the 2D grid into a 1D array for the Linear layers
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    

class Agent:
    def __init__(self):
        self.model = CNN_2048()
        self.target_model = CNN_2048()
        self.target_model.load_state_dict(self.model.state_dict()) # Copy weights
        self.target_model.eval() # Set to evaluation mode
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        # Memory stores past experiences: (state, action, reward, next_state, done)
        self.memory = deque(maxlen=50000)
        
        # Epsilon controls exploration. 
        # 1.0 means 100% random moves. It decays over time as the AI gets smarter.
        self.epsilon = 1.0
        self.learning_rate = 5e-5
        self.epsilon_decay = 1-self.learning_rate
        self.epsilon_min = 0.05
        self.gamma = 0.9 # Discount factor for future rewards

    def preprocess_state(self, state):
        """Prepares a 16-channel one-hot encoded 4x4 grid."""
        # Create an empty tensor of shape (16 channels, 4 height, 4 width)
        tensor_state = torch.zeros((16, 4, 4), dtype=torch.float32)
        
        for i in range(4):
            for j in range(4):
                val = state[i, j]
                if val == 0:
                    # Channel 0 represents empty space
                    tensor_state[0, i, j] = 1.0
                else:
                    # Map the tile value to a channel index (2->1, 4->2, 8->3, etc.)
                    layer = int(np.log2(val))
                    
                    # Cap the maximum tile we track at 32768 (layer 15) to prevent crashes
                    if layer < 16:
                        tensor_state[layer, i, j] = 1.0
                        
        # Output shape: (16, 4, 4)
        return tensor_state

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
        """Trains the neural network using vectorized batching and a Target Network."""
        if len(self.memory) < batch_size:
            return 

        batch = random.sample(self.memory, batch_size)
        
        # 1. Stack the lists of data into batched PyTorch tensors
        states = torch.stack([self.preprocess_state(b[0]) for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([self.preprocess_state(b[3]) for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        # 2. Get current Q-values from the MAIN model
        # .gather() efficiently plucks out the specific Q-values for the actions taken
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 3. Calculate target Q-values using the TARGET model
        with torch.no_grad():
            # Notice we use self.target_model here!
            max_future_q = self.target_model(next_states).max(1)[0]
            
            # Vectorized math: if done=1.0, (1 - dones) becomes 0, cancelling future reward
            target_q = rewards + (self.gamma * max_future_q * (1 - dones))

        # 4. Calculate Loss and Update weights
        # target_q is securely defined right above this, so it will not throw an error
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Slowly reduce exploration (Epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            

def instantiate_and_train():
    # Assuming Game2048Env is defined in your code
    env = Game2048Env()
    agent = Agent()


    episodes = 500
    save_every = 10

    # Load the model if it exists
    for model_nr in range(500,0,-save_every):
        model_path = f"convolutional_model_checkpoint_episode_{model_nr}.pt"
        print("checking", model_path)
        if os.path.exists(model_path):
            agent.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
            break
    else:
        print("No saved model found. Starting with a new model.")

    for episode in range(model_nr, episodes):
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
        # At the end of your episode loop:
        if episode % 10 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            
        print(f"Episode {episode + 1}/{episodes} - Score: {info['score']} - Max Tile: {info['highest_tile']} - Epsilon: {agent.epsilon:.2f}")
        
        # Save model every 10 episodes
        if (episode + 1) % 10 == 0:
            torch.save(agent.model.state_dict(), f"convolutional_model_checkpoint_episode_{episode + 1}.pt")
            print(f"Model saved at episode {episode + 1}")
            
            # Delete the checkpoint from 10 episodes ago
            old_episode = episode + 1 - 10
            if old_episode > 0:
                old_model_path = f"convolutional_model_checkpoint_episode_{old_episode}.pt"
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
                    print(f"Deleted old checkpoint: {old_model_path}")
                    

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