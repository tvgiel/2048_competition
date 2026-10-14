# Participant Guide

How to build, test and submit your 2048 agent. Read the [README](README.md) for
the official rules; this guide is the practical walkthrough.

## 1. What you submit

One file (two at most), named after your nickname:

| Language | File name | You define |
|---|---|---|
| Python | `agent_<name>.py` | `train_and_instantiate()` and `act(agent, state, valid_actions)` |
| Julia | `agent_<name>.jl` | the same two functions |
| C++ | `agent_<name>.cpp` (+ optional header) | `void train_and_instantiate()` and `int act(const int state[4][4], const std::vector<int>& valid_actions)` |

`<name>` is letters and digits only. Actions are `0` = Up, `1` = Right,
`2` = Down, `3` = Left. `act` must return one of `valid_actions`.

Minimal examples (they play randomly; replace with your own idea):

**Python**
```python
import random

def train_and_instantiate():
    return random.Random(0)                  # your trained model goes here

def act(agent, state, valid_actions):
    return agent.choice(valid_actions)       # state is a read-only 4x4 NumPy array
```

**Julia**
```julia
using Random

train_and_instantiate() = MersenneTwister(0)               # your trained model

act(agent, state, valid_actions) = rand(agent, valid_actions)   # state is a Matrix{Int}
```

**C++**
```cpp
#include <random>
#include <vector>

static std::mt19937 rng(0);                  // "agent object" = globals/statics

void train_and_instantiate() {}              // train here and keep results in globals

int act(const int state[4][4], const std::vector<int>& valid_actions) {
    return valid_actions[rng() % valid_actions.size()];
}
```

Important:

- Your file is **imported/included by the harness, not run on its own.** Don't
  start training or play games at import time; put that in `train_and_instantiate()`.
  (Python: guard any demo code with `if __name__ == "__main__":`.)
- The agent must **train itself** inside `train_and_instantiate()`. No
  pre-trained weights loaded from files you did not submit.
- Printing is allowed, but output is discarded into a log.

## 2. Develop and train locally

Use the provided environment for Python or Julia (`game2048.py`, `game2048.jl`,
see `game_api.md`). A quick loop to try your agent without any harness:

```python
from game2048 import Game2048Env
import agent_yourname as m

agent = m.train_and_instantiate()
env = Game2048Env()
state = env.reset()
while not env.done:
    state, reward, done, info = env.step(m.act(agent, state, env.get_available_actions()))
print(env.score, info["highest_tile"])
```

C++ participants do not get an environment class. Port the rules into your own
code for training and search. The rules are short, and `game2048.py` is the
reference.

## 3. Test with the official harness

This runs your agent exactly like the jury does: the same container image, the
same limits, and the same game engine.

**One-time setup** (needs [Docker](https://docs.docker.com/get-docker/); on
Windows use WSL2). Run these from the repository's main folder:

```bash
docker build -t 2048-eval:latest safety
```

The first build downloads Python, Julia and PyTorch and takes a while. After
that it is cached.

**Check your agent:**

```bash
python safety/check_submission.py agent_yourname.py
python safety/check_submission.py agent_yourname.py --games 20      # more games
python safety/check_submission.py my_submission_folder/             # a two-file submission
```

It checks that your file is named correctly and defines the right functions,
compiles it (C++), trains it, plays practice games, and prints:

```
[yourname] game 1/5 score=3180 max=256 moves=270 0.4s
 ...
Training took 12.3s; games averaged 0.41s.
Rough projection for 100 games: 0.9 min (limit 30 min).
PASS  no problems found.
```

The time projection is rough, because better agents play longer games, so
leave a safety margin. Logs, including your program's output and any compiler
messages, are kept in the folder printed at the start (use `--out some_folder`
to choose it).

**No Docker?** `--no-docker` runs your agent directly on your machine with no
isolation. It is fine for quick debugging, but it does not prove your
submission works in the real environment, so do a Docker check before
submitting. You need Python, plus Julia with `JSON3` for Julia agents or `g++`
for C++ agents.

## 4. What the harness does to your agent

- **Seeds:** all agents play the same 100 game seeds, which are secret until the
  event is over.
- **Isolation:** your agent runs in a container with no network, a read-only
  file system (only `/tmp` is writable, 2 GB), no GPU unless announced, 2 CPU
  cores, 4 GB RAM and at most 256 processes.
- **Time:** 30 minutes in total for compiling, training and all 100 games.
  Each move must be returned within 5 s (60 s for the first move).
- **Invalid moves:** returning an action not in `valid_actions`, or raising an
  exception, makes the harness play a random valid move for you.
- **Failures:** a timeout, crash, compile error, training error or garbled
  output stops your agent. Your current game keeps its score, and unplayed
  games score 0.

Result statuses you may see:

| Status | Meaning |
|---|---|
| `finished` | all games played |
| `compile_failed` | C++ did not compile (see `<name>.compile.log`) |
| `train_failed` | `train_and_instantiate()` raised an error or crashed |
| `timeout` | a move took longer than the per-move limit |
| `budget_exceeded` | the 30-minute total ran out |
| `crashed` | your process died (out of memory, segfault, `exit`) |
| `protocol_violation` | output was not what the harness expects |

## 5. Pre-submission checklist

- [ ] File is named `agent_<name>.py`, `.jl` or `.cpp`, and I have at most two files.
- [ ] `python safety/check_submission.py ...` ends with `PASS` using Docker.
- [ ] Projected time for 100 games is well under 30 minutes (including training and compiling).
- [ ] I only use packages in the official image (or agreed an exception with the jury).
- [ ] I load no external weights or data files.
- [ ] Nothing important happens at import time outside the two functions.

## 6. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `Docker image '2048-eval:latest' not found` | Run the build command from step 3. |
| `ModuleNotFoundError` / Julia package missing | The package is not in the official image. Ask the jury before the deadline. |
| `expected exactly: int act(...)` (C++) | The signature must match the one in the README character for character, apart from the parameter names. |
| `compile_failed` | Open `<name>.compile.log`. Only the standard library is available. |
| `timeout` on the first move | Heavy start-up work (loading, JIT) belongs in `train_and_instantiate()`. |
| `Permission denied` / read-only file system | Write only inside `/tmp`. |
| `act() raised an exception` | Look in `<name>.stderr.log` for the traceback. |
| Works with `--no-docker`, fails in Docker | A missing package, a file you didn't submit, or writing outside `/tmp`. |
