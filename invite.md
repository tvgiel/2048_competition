# Introduction

Welcome to the 2048 AI Solving Competition! Your objective is to build a
Reinforcement Learning agent, search algorithm, or heuristic bot capable
of mastering the classic game of 2048. Participants are provided with
the complete, open-source Python environment to develop and train their
models locally before submitting them for final evaluation.

# The Environment

The official competition environment, `Game2048Env`, is given to you.
The exact logic and random generation code used for evaluation are the
same as the ones provided to you.

- **State Space:** A 4x4 matrix representing the game board.

- **Action Space:** Discrete actions:
  `0 (Up), 1 (Right), 2 (Down), 3 (Left)`.

- **Rewards:** The reward for each step is the sum of the merged tiles.
  Invalid moves return a penalty of -1.

- **Dynamics:** Standard 2048 rules apply. After a valid move, a new
  tile spawns in a random empty cell (90% chance of a 2, 10% chance of a
  4).

# Submission API and rules

The submitted code has to be named `agent_<yourname>.py` or
`agent_<yourname>.jl`. Each participant may submit one entry. A submission
may contain at most two files, unless the jury agrees to an exception.

Your submission must provide the following functions so that it can be tested
automatically:

- `train_and_instantiate()`: a function without arguments. When this
  function is called, the model is initialised and trained. returns a
  trained model.

- `act(agent, state, valid_actions)`: a function that takes the trained
  model as an argument. When this function is called, it receives the
  current state of the game and the valid actions. It should return the
  action that the agent wants to take (a number between 0 and 3).

# Submission Limitations

To ensure a standardized and automated evaluation process, all
submissions must strictly adhere to the following limitations:

1.  **Language Constraints:** Agents must be written in **Python**
    (`.py`) or **Julia** (`.jl`).

2.  There is no limit on development or training time before submission. The
  agent must train itself from the provided environment and may not load
  externally trained weights.

3.  **Zero-Argument Execution:** Your submitted file must be fully
    executable by calling the file directly **without any command-line
    arguments**.

    - For Python submissions, the evaluation server will run:
      `python agent_yourname.py`

    - For Julia submissions, the evaluation server will run:
      `julia agent_yourname.jl`

4.  **Evaluation Time:** All 100 games for an agent must be completed within
  30 minutes, including any model initialization or training performed by
  the submission.

5.  **Self-Contained Logic:** Your script must handle its own environment
  setup and training. It may not depend on files, models, or data that are
  not included in the submission.

6.  **Model Size:** The maximum permitted size of a submitted model and its
  included files will be announced by the jury before submissions open.

7.  **Dependencies:** You may use standard data science and machine
    learning libraries. For python these include PyTorch, TensorFlow,
    and everything in anaconda python version 3.19.3. In Julia, this
  includes (F)lux.jl and the other packages listed in the published
  competition environment. If you need a package that is not
  pre-installed, discuss it with the jury before submission and include
  the required dependency declaration (`requirements.txt` for Python or
  `Project.toml` for Julia).

# Evaluation and Scoring

Submissions will be evaluated on a secure server to prevent environment
manipulation.

- **Trials:** Each AI will play 100 games, until the end. Agents will take
  turns playing one game at a time so that live scores and progress graphs
  can be updated during the event.

- **Primary Metric:** Competitors will be ranked based purely on the
  **average final score** across all 100 games.

- **Tie-breaker:** In the event of identical average scores, the maximum
  tile achieved across the 100 games (e.g., reaching 8192 rather than 4096)
  will determine the winner.

- **Logging:** Scores and other agreed evaluation statistics will be logged
  throughout the event and used to update the live graphs.

# Rules and Provided Code

The competition uses the standard rules of 2048: equal adjacent tiles merge
once per move, tiles slide as far as possible, and a new tile appears after a
valid move. A new tile is a 2 with 90% probability or a 4 with 10% probability.
For a general description of the original game rules, see the
[official 2048 game](https://play2048.co/).

The Python and Julia source code for the game itself and for the evaluation
environment will be provided to all participants. The supplied environment is
the authoritative specification for scoring and behavior.

# Event Details

Each submitted agent must have a nickname. The nickname will be used to
identify the model in the live stream and in the score display.

Good luck, and may the best AI win!
