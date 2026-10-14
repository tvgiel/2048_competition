# Introduction

Welcome to the 2048 AI Solving Competition! Your objective is to build a
Reinforcement Learning agent, search algorithm, or heuristic bot capable of
mastering the classic game of 2048. Participants are provided with the complete,
open-source Python/Julia environment and the evaluation harness, so they can
develop, train and test their agents locally before submitting them for final
evaluation. Agents can be written in **Python, Julia or C++**.

New here? Start with the step-by-step [Participant Guide](PARTICIPANT_GUIDE.md).

# The Environment

The official competition environment, `Game2048Env`, is given to you. The exact logic and random generation code used for evaluation are the same as the ones provided to you.

- **State Space:** A 4x4 matrix representing the game board (`0` = empty cell).

- **Action Space:** Discrete actions:
  `0 (Up), 1 (Right), 2 (Down), 3 (Left)`.

- **Rewards:** The reward for each step is the sum of the merged tiles.
  Invalid moves return a penalty of -1.

- **Dynamics:** Standard 2048 rules apply. After a valid move, a new tile
  spawns in a random empty cell (90% chance of a 2, 10% chance of a 4).

The environment is provided in Python (`game2048.py`) and Julia (`game2048.jl`).
See `game_api.md` for its methods. C++ agents do not use the environment
directly; they receive the board from the evaluation harness (see below), and
can port the game logic themselves for training and search.

# Languages and the Submission API

The submitted file has to be named `agent_<yourname>.py`, `agent_<yourname>.jl` or `agent_<yourname>.cpp`. 

Your submission must provide the following two functions so that it can be tested automatically. The harness imports (Python), includes (Julia) or links (C++) your file; it does not run it as a standalone program, so do not rely on
running anything at import time other than defining your functions and constants.

## Python and Julia

- `train_and_instantiate()`: a function without arguments. When called, the model is initialised and trained. It returns the trained model (the "agent").

- `act(agent, state, valid_actions)`: takes the object returned by
  `train_and_instantiate()`, the current board (a read-only 4x4 NumPy array in  Python, a `Matrix{Int}` in Julia) and the list of valid actions. It returns the action to play, a number between 0 and 3.

## C++

C++17, compiled with `g++ -O3 -std=c++17 -pthread` together with a
harness-provided `main()`. Standard library only. Define exactly:

```cpp
#include <vector>

void train_and_instantiate();
int  act(const int state[4][4], const std::vector<int>& valid_actions);
```

There is no agent object: keep your trained model in global or `static`
variables. `state[row][col]` holds the tile values (`0` = empty). Anything you
print to `stdout` is discarded (it goes to a log), so it cannot break the
harness.

# Submission Limitations

To ensure a standardized and automated evaluation process, all submissions
must strictly adhere to the following limitations:

1.  **Language Constraints:** Agents must be written in **Python** (`.py`),
    **Julia** (`.jl`) or **C++** (`.cpp`).

2.  **Files:** A submission contains at most two files (not counting a
    dependency declaration, see 7), unless the jury agrees to an exception. For
    C++, the second file may be a header (`.h`/`.hpp`) or another `.cpp`. Each
    participant may submit one entry.

3.  **Training:** There is no limit on development or training time *before*
    submission. The agent must then load the trained model. If training takes long, see rule 4.

4.  **Evaluation Time:** All 100 games for an agent must be completed within
    30 minutes. This budget includes everything the agent does: C++
    compilation, loading and initialising the model, any training performed by `train_and_instantiate()`, and all of its moves. Each individual move
    must be returned within 5 seconds (60 seconds for the very first move, to
    allow for warm-up).

5.  **Self-Contained Logic:** Your submission may not depend on files, models
    or data that are not included in it. At evaluation the agent has **no
    network access**, and the file system is read-only except for a temporary
    directory (`/tmp`).

6.  **Model Size:** The maximum permitted size of a submitted model and its
    included files will be announced by the jury before submissions open.

7.  **Dependencies:** Agents run in the official evaluation Docker image
    (provided in this repository, see `safety/Dockerfile`). It contains
    Python 3.12 with NumPy, SciPy, pandas, scikit-learn, Numba and PyTorch
    (CPU) and optionally TensorFlow; Julia with Flux.jl, StatsBase.jl and
    Distributions.jl; and the GNU C++ compiler. The jury will confirm the final
    package list before submissions open. If you need a package that is not
    installed, discuss it with the jury before submission and include the
    required dependency declaration (`requirements.txt` for Python or
    `Project.toml` for Julia).

8.  **Resources:** Each agent runs alone on the evaluation machine, limited to
    2 CPU cores, 4 GB of memory and 256 processes (the jury will announce the
    final values and whether a GPU is available).

# Evaluation and Scoring

Submissions are evaluated on a secure machine. Each agent runs inside an
isolated container, and the game itself is run by the harness, outside of the
agent's reach. The agent only receives copies of the board.

- **Trials:** Each AI plays 100 games, until the end. Every agent plays the
  same 100 game seeds. Agents take turns playing one game at a time so that
  live scores and progress graphs can be updated during the event.

- **Primary Metric:** Competitors are ranked purely on the **average final
  score** across all 100 games.

- **Tie-breaker:** In the event of identical average scores, the maximum tile achieved across the 100 games (e.g., reaching 8192 rather than 4096) determines the winner. If there is still a tie, the agent that needed the least time to finish the 100 games wins. If there is still a tie, the smallest submission file wins.

- **Invalid moves:** If `act()` returns an action that is not in
  `valid_actions`, or raises an exception, the harness plays a random valid
  move instead. This costs you the move, so only return valid actions.

- **Failures:** If your agent exceeds a time limit, crashes, fails to compile
  or to train, or writes malformed output, it is stopped. The game in progress
  keeps its current score, and all games not yet played score 0.

- **Logging:** Scores and other agreed evaluation statistics are logged
  throughout the event and used to update the live graphs. After the event,
  the game seeds and move sequences are published so results can be replayed.

# Test your submission before you submit

This repository contains the same harness, image and limits that the jury uses.
`check_submission.py` runs your agent on a few practice games and reports
problems. A submission that does not pass it may be rejected. In short:

```bash
docker build -t 2048-eval:latest safety
python safety/check_submission.py agent_yourname.py
```

The [Participant Guide](PARTICIPANT_GUIDE.md) explains this in detail.

# Rules and Provided Code

The competition uses the standard rules of 2048: equal adjacent tiles merge
once per move, tiles slide as far as possible, and a new tile appears after a
valid move. A new tile is a 2 with 90% probability or a 4 with 10% probability.
For a general description of the original game rules, see the
[official 2048 game](https://play2048.co/).

The Python and Julia source code for the game itself, the evaluation harness
and the evaluation image are provided to all participants. The supplied
environment is the authoritative specification for scoring and behavior.

# Event Details

Each submitted agent must have a nickname. The nickname will be used to
identify the model in the live stream and in the score display. It is the
`<yourname>` part of your file name.

Good luck, and may the best AI win!
