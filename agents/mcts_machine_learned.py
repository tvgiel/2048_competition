import copy
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game2048 import Game2048Env
from mcts_AI import get_best_move_mc


def _state_to_board(state: Any) -> np.ndarray:
    board = np.array(state)
    if board.size == 16 and board.shape != (4, 4):
        board = board.reshape(4, 4)
    return board.astype(np.float32)


def _board_features(board: np.ndarray) -> np.ndarray:
    # log2 transform for non-zero tiles; scale down for stability
    x = board.flatten().astype(np.float32)
    feats = np.zeros_like(x, dtype=np.float32)
    nz = x > 0
    feats[nz] = np.log2(x[nz]) / 16.0
    return feats  # (16,)


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int = 16, hidden: int = 128, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _collect_mcts_dataset(
    games: int,
    max_moves: int,
    teacher_sims: int,
    teacher_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []

    for _ in range(games):
        env = Game2048Env()
        env.reset()

        moves = 0
        while not env.done and moves < max_moves:
            valid = env.get_available_actions()
            if not valid:
                break

            # Teacher action from your MCTS agent
            action = get_best_move_mc(
                env=env,
                simulations_per_action=teacher_sims,
                max_steps=teacher_steps,
                candidate_actions=valid,
                use_multiprocessing=False,
            )

            X.append(_board_features(env.board))
            y.append(int(action))

            env.step(int(action))
            moves += 1

    if len(X) == 0:
        return np.zeros((0, 16), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def _train_policy(
    model: PolicyNet,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> None:
    if len(X) == 0:
        return

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    n = X_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_and_instantiate():
    """
    Required API.
    Trains a PyTorch policy network to imitate MCTS, then returns the agent.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyNet().to(device)

    # Moderate defaults for runtime; increase for stronger policy.
    X, y = _collect_mcts_dataset(
        games=24,
        max_moves=200,
        teacher_sims=25,
        teacher_steps=60,
    )

    _train_policy(
        model=model,
        X=X,
        y=y,
        device=device,
        epochs=25,
        batch_size=128,
        lr=1e-3,
    )

    model.eval()

    return {
        "template_env": Game2048Env(),
        "policy_model": model,
        "device": device,
        # NN-guided MCTS inference settings:
        "mcts_top_k": 2,   # top-k NN actions passed to MCTS
        "mcts_sims": 40,
        "mcts_steps": 80,
    }


def act(agent, state, valid_actions):
    """
    Required API.
    Uses NN-guided MCTS: NN ranks actions, MCTS searches top-k.
    """
    if not valid_actions:
        return -1

    board = _state_to_board(state)
    feats = _board_features(board)[None, :]  # (1,16)

    model: PolicyNet = agent["policy_model"]
    device: torch.device = agent["device"]

    with torch.no_grad():
        x = torch.tensor(feats, dtype=torch.float32, device=device)
        logits = model(x)[0].detach().cpu().numpy()

    ranked = sorted(valid_actions, key=lambda a: logits[a], reverse=True)

    k = int(agent.get("mcts_top_k", 2))
    k = max(1, min(k, len(ranked)))
    candidates = ranked[:k]

    if len(candidates) == 1:
        return int(candidates[0])

    env = copy.deepcopy(agent["template_env"])
    env.board = board.astype(int)
    env.done = False

    return int(
        get_best_move_mc(
            env=env,
            simulations_per_action=agent.get("mcts_sims", 40),
            max_steps=agent.get("mcts_steps", 80),
            candidate_actions=candidates,
            use_multiprocessing=False,
        )
    )