import numpy as np
import random

class Game2048Env:
    """
    2048 Game Environment with a standard AI Trainer API.
    Actions: 0 (Up), 1 (Right), 2 (Down), 3 (Left)
    """
    def __init__(self, size=4):
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.done = False

    def reset(self):
        """Resets the environment and returns the initial state."""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()
        return self.get_state()

    def step(self, action):
        """
        Takes an action and advances the environment state.
        action: 
            - 0 = up
            - 1 = right
            - 2 = down
            - 3 = left
        Returns: (state, reward, done, info)
        """
        if self.done:
            return self.get_state(), 0, self.done, {"error": "Game is already over"}

        original_board = self.board.copy()
        reward = 0

        # 0: Up (k=1), 1: Right (k=2), 2: Down (k=3), 3: Left (k=0)
        rotation_k = (action + 1) % 4
        
        self.board = np.rot90(self.board, k=rotation_k)

        for i in range(self.size):
            self.board[i], row_reward = self._slide_and_merge(self.board[i])
            reward += row_reward

        # Rotate the board back to its original orientation
        self.board = np.rot90(self.board, k=-rotation_k)

        # If the board changed, the move was valid
        if not np.array_equal(original_board, self.board):
            self.score += reward
            self._spawn_tile()
        else:
            reward = -1 

        self.done = self._check_game_over()
        
        info = {
            "score": self.score,
            "highest_tile": np.max(self.board),
            "valid_move": not np.array_equal(original_board, self.board)
        }
        
        return self.get_state(), reward, self.done, info

    def get_available_actions(self):
        """Returns a list of actions that will actually change the board state."""
        actions = []
        for action in range(4):
            # Apply the exact same rotation logic to test actions
            rotation_k = (action + 1) % 4
            test_board = np.rot90(self.board.copy(), k=rotation_k)
            changed = False
            
            for i in range(self.size):
                new_row, _ = self._slide_and_merge(test_board[i])
                if not np.array_equal(test_board[i], new_row):
                    changed = True
                    break
            if changed:
                actions.append(action)
        return actions
        

    def get_state(self):
        """Returns the current board. Using log2 helps Neural Networks process the data."""
        return self.board.copy()

    def _slide_and_merge(self, row):
        """Core logic: slides non-zero elements left and merges adjacent duplicates."""
        # 1. Slide all non-zero numbers to the left
        non_zero = [x for x in row if x != 0]
        reward = 0
        
        # 2. Merge identical adjacent numbers
        merged_row = []
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i < len(non_zero) - 1 and non_zero[i] == non_zero[i+1]:
                merged_val = non_zero[i] * 2
                merged_row.append(merged_val)
                reward += merged_val
                skip = True
            else:
                merged_row.append(non_zero[i])
                
        # 3. Pad the rest of the row with zeros
        merged_row.extend([0] * (self.size - len(merged_row)))
        return np.array(merged_row), reward

    def _spawn_tile(self):
        """Spawns a 2 (90% chance) or 4 (10% chance) in a random empty cell."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r, c] = 4 if random.random() < 0.1 else 2

    def _check_game_over(self):
        """Game is over if there are no empty cells and no possible merges."""
        if np.any(self.board == 0):
            return False
            
        # Check for horizontal merges
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
                    
        # Check for vertical merges
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
                    
        return True

    def render(self):
        """Prints the board to the console for human debugging."""
        print("-" * 25)
        for row in self.board:
            print("|" + "|".join(f"{num:5}" if num != 0 else "     " for num in row) + "|")
        print("-" * 25)
        print(f"Score: {self.score} | Max Tile: {np.max(self.board)}\n")
        
 