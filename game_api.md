# 2048 API (Short Overview)

This project exposes two APIs:

## 1) Game Environment API (`Game2048Env`)
`Game2048Env` is the 2048 simulator. It manages the board, score, tile spawning, merges, and game-over checks.

- **Actions:** `0=Up`, `1=Right`, `2=Down`, `3=Left`
- **`reset()`**: starts a new game and returns the initial board state.
- **`step(action)`**: applies one move and returns  
  `(state, reward, done, info)`.
- **`get_available_actions()`**: returns only moves that change the board.
Returns a list of valid action indices (0-3).
- **`get_state()`**: returns a copy of the current board. 
Python: Returns a 4x4 numpy array with tile values (0 for empty).µ
Julia: Returns a 4x4 matrix with tile values (0 for empty).
- **`render()`**: prints the board for debugging.

## 2) Agent API (competition interface)
Each AI agent file must implement:

- **`train_and_instantiate()`**: creates (and optionally trains) the agent; returns the agent object.
- **`act(agent, state, valid_actions)`**: chooses and returns one action (`0` to `3`) using the current state and valid actions.

In short, the environment API runs the game, and the agent API plugs an AI policy into that game loop.