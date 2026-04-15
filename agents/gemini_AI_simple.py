import numpy as np
import time
from game2048 import Game2048Env

# Snake-shaped weight matrix to encourage building in the top-left corner
WEIGHT_MATRIX = np.array([
    [2**15, 2**14, 2**13, 2**12],
    [2**8,  2**9,  2**10, 2**11],
    [2**7,  2**6,  2**5,  2**4],
    [2**0,  2**1,  2**2,  2**3]
], dtype=float)

def evaluate_board(board):
    """Calculate the heuristic score of a board."""
    return np.sum(board * WEIGHT_MATRIX)

def simulate_step(board, action):
    """
    Simulates a board step to check future states without instantiating Game2048Env.
    action: 0 (Up), 1 (Right), 2 (Down), 3 (Left)
    Returns: new_board, reward, is_changed
    """
    new_board = np.copy(board)
    reward = 0
    size = new_board.shape[0]
    
    rotation_k = (action + 1) % 4
    new_board = np.rot90(new_board, k=rotation_k)
    
    is_changed = False
    for i in range(size):
        row = new_board[i]
        non_zero = [x for x in row if x != 0]
        
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
                
        merged_row.extend([0] * (size - len(merged_row)))
        new_row = np.array(merged_row)
        
        if not np.array_equal(row, new_row):
            is_changed = True
        new_board[i] = new_row

    new_board = np.rot90(new_board, k=-rotation_k)
    return new_board, reward, is_changed

def expectimax(board, depth, is_player):
    if depth == 0:
        return evaluate_board(board)
        
    if is_player:
        best_score = -float('inf')
        moved = False
        for action in range(4):
            new_board, reward, is_changed = simulate_step(board, action)
            if is_changed:
                moved = True
                score = reward + expectimax(new_board, depth - 1, False)
                best_score = max(best_score, score)
                
        if not moved:
            return evaluate_board(board)
        return best_score
        
    else:
        # Chance node (game spawning 2 or 4)
        empty_cells = list(zip(*np.where(board == 0)))
        if not empty_cells:
            return evaluate_board(board)
            
        expected_score = 0
        num_empty = len(empty_cells)
        prob_2 = 0.9 / num_empty
        prob_4 = 0.1 / num_empty
        
        for r, c in empty_cells:
            board[r, c] = 2
            expected_score += prob_2 * expectimax(board, depth - 1, True)
            
            board[r, c] = 4
            expected_score += prob_4 * expectimax(board, depth - 1, True)
            
            board[r, c] = 0 # backtrack
            
        return expected_score

def get_best_move(env, depth=3):
    """Finds the best move using Expectimax search."""
    best_score = -float('inf')
    best_action = -1
    board = env.get_state()
    
    for action in range(4):
        new_board, reward, is_changed = simulate_step(board, action)
        if is_changed:
            score = reward + expectimax(new_board, depth - 1, False)
            if score > best_score:
                best_score = score
                best_action = action
                
    return best_action

if __name__ == "__main__":
    env = Game2048Env()
    env.reset()
    
    action_map = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    move_count = 0
    
    print("Starting 2048 AI (Expectimax)...")
    start_time = time.time()
    
    while not env.done:
        # Dynamic depth: search deeper when the board has fewer empty cells
        empty_count = len(np.where(env.board == 0)[0])
        current_depth = 3 if empty_count > 4 else 4
        
        best_action = get_best_move(env, depth=current_depth)
        
        if best_action == -1:
            break
            
        _, reward, done, info = env.step(best_action)
        move_count += 1
        
        if move_count % 50 == 0 or done:
            print(f"Move {move_count}: played {action_map[best_action]}")
            env.render()
            
    total_time = time.time() - start_time
    print("="*30)
    print("GAME OVER")
    print(f"Final Score: {env.score}")
    print(f"Highest Tile: {np.max(env.board)}")
    print(f"Total Moves: {move_count}")
    print(f"Time Taken: {total_time:.2f} seconds")
    print("="*30)
