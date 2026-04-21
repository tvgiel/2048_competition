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
`agent_<yourname>.jl`. It needs the same main functions as provided to
you [TBD HOW]{style="color: red"}. Your submission needs some functions
for us to test it easily:

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

2.  Your agent may take at most 1 hour to train on a simple, CPU-only
    laptop. and [30 minutes, tbd]{style="color: red"} to execute.
    Training will be stopped at 1 hour, and the resulting agent will be
    used for scoring.

3.  **Zero-Argument Execution:** Your submitted file must be fully
    executable by calling the file directly **without any command-line
    arguments**.

    - For Python submissions, the evaluation server will run:
      `python agent_yourname.py`

    - For Julia submissions, the evaluation server will run:
      `julia agent_yourname.jl`

4.  **Self-Contained Logic:** Your script must handle its own
    environment setup. It cannot load any pre-trained weights.

5.  **Dependencies:** You may use standard data science and machine
    learning libraries. For python these include PyTorch, TensorFlow,
    and everything in anaconda python version 3.19.3. In Julia, this
    includes Flux.jl EN NOG ANDERE TE BEPALEN . A list of pre-installed
    environment packages will be provided. If you have custom
    dependencies, include a `requirements.txt` or `Project.toml`.

# Evaluation and Scoring

Submissions will be evaluated on a secure server to prevent environment
manipulation.

- **Trials:** Each AI will play 10 games, until the end.

- **Primary Metric:** Competitors will be ranked based purely on the
  **average final score** across all 10 games.

- **Tie-breaker:** In the event of identical average scores, the maximum
  tile achieved across the 100 episodes (e.g., reaching 8192 vs 4096)
  will determine the winner.

# TODO

\- 1 entry per person - geen limiet om traintijd - 100 games spelen in
max 30 minuten - model maximaal aantal MB - indien speciale packages,
overleg met Jury - zelf trainen - gemiddelde alle scores - verwijs naar
regels van het spel - code van spel zelf geven (python en julia) - code
van test-environment geven (python en julia) - maximum 2 files indienen
(tenzij in overleg) - elk om beurt 1 spel spelen. Zo live grafieken
hebben om te updaten. (log as voor de punten) - bijnamen verzinnen voor
de modellen

Good luck, and may the best AI win!
