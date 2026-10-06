# 2048 API (Short Overview)

This project exposes two APIs:

## 1) Game Environment API (`Game2048Env`)
`Game2048Env` is the 2048 simulator. It manages the board, score, tile spawning, merges, and game-over checks.

The default board size is 4x4. Both implementations also accept an optional
`size` argument when constructing an environment.

- **Actions:** `0=Up`, `1=Right`, `2=Down`, `3=Left`
- **Reset:** starts a new game, places two initial tiles, and returns the
  initial board state.
- **Step:** applies one move and returns `(state, reward, done, info)`. A
  valid move spawns one new tile. An invalid move returns a reward of `-1`.
- **Available actions:** returns only moves that would change the board. The
  result contains valid action indices from `0` to `3`.
- **State:** returns a copy of the current board, with `0` representing an
  empty cell. The default result is a 4x4 array or matrix; its dimensions
  match the configured `size`.
- **Render:** prints the board, score, and highest tile for debugging.

The Python environment uses these methods:

```python
env = Game2048Env(size=4)
state = env.reset()
state, reward, done, info = env.step(action)
valid_actions = env.get_available_actions()
state = env.get_state()
env.render()
```

The Julia environment uses these methods:

```julia
env = Game2048Env(4)
state = reset!(env)
state, reward, done, info = step!(env, action)
valid_actions = get_available_actions(env)
state = get_state(env)
render(env)
```

Julia uses `reset!` and `step!` because those functions mutate the
environment. The Python versions are `reset` and `step`.

## 2) Agent API (competition interface)
Each AI agent file must implement:

- **`train_and_instantiate()`**: creates (and optionally trains) the agent; returns the agent object.
- **`act(agent, state, valid_actions)`**: chooses and returns one action (`0` to `3`) using the current state and valid actions.

In short, the environment API runs the game, and the agent API plugs an AI policy into that game loop.