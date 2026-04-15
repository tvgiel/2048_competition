import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from game2048 import Game2048Env
from gemini_AI_simple import get_best_move as get_best_move_expectimax
from mcts_AI import get_best_move_mc
from one_direction_AI import SimplePriorityAI

def play_game_priority(_):
    env = Game2048Env()
    env.reset()
    ai = SimplePriorityAI()
    while not env.done:
        action = ai.get_best_move(env)
        if action == -1: break
        env.step(action)
    return env.score

def play_game_expectimax(_):
    env = Game2048Env()
    env.reset()
    while not env.done:
        # Reduced depth for bulk testing to execute in a reasonable time
        action = get_best_move_expectimax(env, depth=2)
        if action == -1: break
        env.step(action)
    return env.score

def play_game_mcts(_):
    env = Game2048Env()
    env.reset()
    while not env.done:
        # Reduced simulations for bulk testing
        action = get_best_move_mc(env, simulations_per_action=20, max_steps=50)
        if action == -1: break
        env.step(action)
    return env.score

if __name__ == '__main__':
    num_games = 100
    results = {}
    
    print(f"--- Starting Evaluation ({num_games} games per AI) ---")
    
    # 1. Simple Priority AI
    print("\n1. Running Simple Priority AI...")
    start = time.time()
    # Runs fast enough sequentially
    scores_priority = [play_game_priority(i) for i in range(num_games)]
    results['Priority AI'] = scores_priority
    print(f"Done in {time.time()-start:.2f}s")
    print(f"Avg Score: {np.mean(scores_priority):.2f} ± {np.std(scores_priority):.2f}")
    
    # 2. Expectimax AI
    print("\n2. Running Expectimax AI (Depth=2)...")
    start = time.time()
    # Execute in parallel to save time
    with mp.Pool(mp.cpu_count()) as pool:
        scores_expectimax = pool.map(play_game_expectimax, range(num_games))
    results['Expectimax (d=2)'] = scores_expectimax
    print(f"Done in {time.time()-start:.2f}s")
    print(f"Avg Score: {np.mean(scores_expectimax):.2f} ± {np.std(scores_expectimax):.2f}")
    
    # 3. MCTS AI
    print("\n3. Running MCTS AI (Sims=20)...")
    # MCTS internally uses mp.Pool for actions, so we simulate games sequentially to avoid daemon issues
    start = time.time()
    scores_mcts = [play_game_mcts(i) for i in range(num_games)]
    results['MCTS (sims=20)'] = scores_mcts
    print(f"Done in {time.time()-start:.2f}s")
    print(f"Avg Score: {np.mean(scores_mcts):.2f} ± {np.std(scores_mcts):.2f}")

    # --- Plotting the Results ---
    print("\nGenerating boxplot...")
    labels = list(results.keys())
    data = [results[label] for label in labels]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.title(f'2048 AI Performance Comparison ({num_games} Games Each)')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save and show the plot
    plt.savefig('ai_comparison_boxplot.png')
    plt.show()




