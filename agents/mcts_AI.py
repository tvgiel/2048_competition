import copy
import multiprocessing as mp
import numpy as np
import random
import time
from typing import Any, Dict, Iterable, Optional
from game2048 import Game2048Env

def simulate_random_game(env, max_steps=100):
    """Plays purely random moves until the game is over or max_steps is reached."""
    current_env = copy.deepcopy(env)
    score = 0
    steps = 0
    
    while not current_env.done and steps < max_steps:
        actions = current_env.get_available_actions()
        if not actions:
            break
            
        action = random.choice(actions)
        _, reward, done, _ = current_env.step(action)
        score += reward
        steps += 1
        
    # Heuristic based on max tile and layout
    max_tile = np.max(current_env.board)
    empty_cells = len(np.where(current_env.board == 0)[0])
    
    return score + (max_tile * 10) + (empty_cells * 20)

def evaluate_action(args):
    """Worker function to simulate multiple games for a single action."""
    env, action, num_simulations, max_steps = args
    total_score = 0
    
    # Take the first step
    next_env = copy.deepcopy(env)
    _, initial_reward, done, _ = next_env.step(action)
    
    if done:
        return initial_reward
        
    for _ in range(num_simulations):
        score = simulate_random_game(next_env, max_steps)
        total_score += score
        
    return initial_reward + (total_score / num_simulations)

def get_best_move_mc(env, simulations_per_action=100, max_steps=100):
    """Finds the best move using Monte Carlo Tree Search executed in parallel"""
    actions = env.get_available_actions()
    if not actions:
        return -1
        
    if len(actions) == 1:
        return actions[0]
        
    pool_args = [(env, action, simulations_per_action, max_steps) for action in actions]
    
    # Utilize Python multiprocessing to map across all available CPU cores!
    with mp.Pool(processes=min(len(actions), mp.cpu_count())) as pool:
        scores = pool.map(evaluate_action, pool_args)
        
    best_idx = np.argmax(scores)
    return actions[best_idx]

def get_best_move_mc(
    env,
    simulations_per_action=100,
    max_steps=100,
    candidate_actions: Optional[Iterable[int]] = None,
    use_multiprocessing: bool = True
):
    """Finds the best move using Monte Carlo rollouts (optionally parallel)."""
    actions = list(candidate_actions) if candidate_actions is not None else env.get_available_actions()
    if not actions:
        return -1

    if len(actions) == 1:
        return actions[0]

    pool_args = [(env, action, simulations_per_action, max_steps) for action in actions]

    if use_multiprocessing and len(actions) > 1:
        with mp.Pool(processes=min(len(actions), mp.cpu_count())) as pool:
            scores = pool.map(evaluate_action, pool_args)
    else:
        scores = [evaluate_action(args) for args in pool_args]

    best_idx = int(np.argmax(scores))
    return actions[best_idx]

def _state_to_env(template_env: Game2048Env, state: Any) -> Game2048Env:
    """Builds an environment from a raw state array or returns a copied env if already env-like."""
    if hasattr(state, "step") and hasattr(state, "get_available_actions") and hasattr(state, "board"):
        return copy.deepcopy(state)

    env = copy.deepcopy(template_env)
    board = np.array(state)

    if board.size == 16 and board.shape != (4, 4):
        board = board.reshape(4, 4)

    env.board = board.astype(int)
    env.done = False
    return env

def train_and_instantiate():
    """
    Required API.
    MCTS is planning-based, so there is no offline training phase.
    Returns a configured agent object.
    """
    return {
        "template_env": Game2048Env(),
        "simulations_per_action": 75,
        "max_steps": 100,
        "use_multiprocessing": False,  # safer default for evaluation harnesses
    }

def act(agent, state, valid_actions):
    """
    Required API.
    Chooses an action in {0,1,2,3} using Monte Carlo rollouts.
    """
    if not valid_actions:
        return -1

    env = _state_to_env(agent["template_env"], state)

    return get_best_move_mc(
        env=env,
        simulations_per_action=agent.get("simulations_per_action", 75),
        max_steps=agent.get("max_steps", 100),
        candidate_actions=valid_actions,
        use_multiprocessing=agent.get("use_multiprocessing", False),
    )

if __name__ == "__main__":
    
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
    
    
    env = Game2048Env()
    start_state = env.reset()
    
    print("Starting 2048 AI (Parallel Monte Carlo Tree Search)")
    print(f"Detected {mp.cpu_count()} CPU cores. Optimized for CPU processing.")
    
    moves = 0
    start_time = time.time()
    action_map = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    
    while not env.done:
        # Increase simulations as the board gets harder
        empty_count = len(np.where(env.board == 0)[0])
        sims = 150 if empty_count < 6 else 75
        
        best_action = get_best_move_mc(env, simulations_per_action=sims, max_steps=100)
        
        if best_action == -1:
             break
             
        _, reward, done, info = env.step(best_action)
        moves += 1
        
        if moves % 1 == 0 or done:
            print(f"Move {moves} | Played {action_map[best_action]} | Max tile {np.max(env.board)} | Score {env.score}")
            env.render()
            
    total_time = time.time() - start_time
    print("==============================")
    print("GAME OVER")
    print(f"Final Score: {env.score}")
    print(f"Highest Tile: {np.max(env.board)}")
    print(f"Total Moves: {moves}")
    print(f"Time Taken: {total_time:.2f} seconds")
    print("==============================")


