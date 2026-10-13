#!/usr/bin/env python3
"""
check_submission.py - participants run this before submitting.

It runs YOUR agent through the same harness, container image and limits the
organisers use, on a few practice games, and reports problems.

    python check_submission.py agent_alice.py
    python check_submission.py my_submission_dir/ --games 10
    python check_submission.py agent_alice.py --no-docker     # if you have no Docker (no sandbox!)

Exit code 0 = no problems found, 1 = problems found.
"""
from __future__ import annotations

import argparse
import ast
import re
import statistics
import sys
import tempfile
from pathlib import Path

import harness


def static_check(entry: Path, lang: str):
    """Cheap checks that the required functions exist with sane signatures."""
    problems = []
    text = entry.read_text(errors="replace")
    if lang == "python":
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            return [f"syntax error: {e}"]
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        f = funcs.get("train_and_instantiate")
        if f is None:
            problems.append("train_and_instantiate() is not defined at top level")
        elif len(f.args.args) - len(f.args.defaults) > 0:
            problems.append("train_and_instantiate() must be callable without arguments")
        f = funcs.get("act")
        if f is None:
            problems.append("act(agent, state, valid_actions) is not defined at top level")
        elif len(f.args.args) < 3 and not f.args.vararg:
            problems.append("act() must accept three arguments: (agent, state, valid_actions)")
    else:
        if not re.search(r"function\s+train_and_instantiate\s*\(\s*\)|^\s*train_and_instantiate\s*\(\s*\)\s*=",
                         text, re.M):
            problems.append("train_and_instantiate() with no arguments not found")
        if not re.search(r"function\s+act\s*\(|^\s*act\s*\([^)]*\)\s*=", text, re.M):
            problems.append("act(agent, state, valid_actions) not found")
    return problems


def main():
    d = harness.Config()
    ap = argparse.ArgumentParser(description="Check a 2048 submission against the official harness.")
    ap.add_argument("submission", help="agent_<name>.py / .jl, or a directory containing it")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1, help="practice seed (not the official one)")
    ap.add_argument("--image", default=d.image)
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--total-budget", type=float, default=d.total_budget)
    ap.add_argument("--move-timeout", type=float, default=d.move_timeout)
    ap.add_argument("--cpus", type=int, default=d.cpus)
    ap.add_argument("--memory", default=d.memory)
    ap.add_argument("--out", default=None, help="keep logs in this directory (default: temp dir)")
    a = ap.parse_args()

    cfg = harness.Config(image=a.image, use_docker=not a.no_docker, games=a.games, master_seed=a.seed,
                         total_budget=a.total_budget, move_timeout=a.move_timeout,
                         cpus=a.cpus, memory=a.memory)
    problems, warnings = [], []

    # 1. Structure and static checks
    try:
        stage, entry, lang, nick = harness.stage_submission(a.submission, cfg.max_files, cfg.max_mb)
    except harness.SubmissionError as e:
        print(f"FAIL  submission rejected: {e}")
        return 1
    static = static_check(stage / entry, lang)
    harness.shutil.rmtree(stage, ignore_errors=True)
    problems += static
    if static:
        for p in static:
            print(f"FAIL  {p}")
        return 1
    print(f"OK    {entry} looks like a valid {lang} submission (nickname '{nick}')")

    # 2. Run it for real
    outdir = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="check_"))
    print(f"Running {a.games} practice games (logs: {outdir}) ...\n")
    runs, rejected = harness.run_competition(cfg, [a.submission], outdir, log=lambda s: print(s, flush=True))
    r = runs[0]
    print()

    if r.status not in ("finished",):
        problems.append(f"run ended with status '{r.status}'. {r.detail[:300]}")
    if any(g["status"] != "ok" for g in r.games):
        problems.append("at least one game did not finish normally (see status above)")
    illegal = sum(g["illegal"] for g in r.games)
    errors = sum(g["errors"] for g in r.games)
    if illegal:
        warnings.append(f"{illegal} illegal move(s): the harness replaced them with random valid moves "
                        "(or forfeited the game). Only return actions from valid_actions.")
    if errors:
        problems.append(f"act() raised an exception {errors} time(s); see {r.stderr_path}")

    # 3. Rough time projection for the official 100 games
    if r.games:
        per_game = statistics.fmean(g["seconds"] for g in r.games)
        projected = r.train_seconds + 100 * per_game
        print(f"Training took {r.train_seconds:.1f}s; games averaged {per_game:.2f}s.")
        print(f"Rough projection for 100 games: {projected / 60:.1f} min "
              f"(limit {cfg.total_budget / 60:.0f} min). Longer-lived (better) games take longer, "
              "so leave a safety margin.")
        if projected > cfg.total_budget:
            problems.append("projected total time exceeds the time limit")
        elif projected > 0.7 * cfg.total_budget:
            warnings.append("projected time is above 70% of the limit; the official machine may be slower")

    for w in warnings:
        print(f"WARN  {w}")
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print("PASS  no problems found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
