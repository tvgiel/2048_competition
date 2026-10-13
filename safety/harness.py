#!/usr/bin/env python3
"""
harness.py - trusted evaluation harness for the 2048 competition.

The harness owns the real Game2048Env, the scoring and the logs. Every agent
runs in its own Docker container (or, with --no-docker, a plain subprocess for
local debugging ONLY) behind shim.py / shim.jl and only ever sees copies of the
board over a JSON-lines pipe.

Examples
    python harness.py submissions/agent_alice.py submissions/bob/
    python harness.py --master-seed 987654 --games 100 submissions/*
    python harness.py --mode interleaved submissions/*      # agents take turns, game by game
    python harness.py --no-docker --games 3 agent_test.py   # debugging, NO sandbox

A submission is a single agent_<name>.py / agent_<name>.jl file, or a
directory containing exactly one such file plus (at most) one helper file.

Policy (edit the Config defaults below, and publish them in the rules)
    * Every agent plays the same game seeds, derived from --master-seed.
    * illegal move        -> a random valid move is played instead (--illegal forfeit
                             ends the game with its current score instead).
    * exception in act()  -> counted as an illegal move.
    * move timeout, crash, protocol violation, or exhausting the total budget
                          -> the current game keeps its score, the agent is
                             stopped, and all unplayed games score 0.
    * The total budget (training + all moves) is measured per agent as the time
      the agent was busy, so interleaved mode does not penalise waiting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from game2048 import Game2048Env

HERE = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^agent_([A-Za-z0-9]+)\.(py|jl)$")
DEP_FILES = {"requirements.txt", "Project.toml", "Manifest.toml"}  # not counted towards max_files
MAX_LINE = 64 * 1024            # longest accepted reply line from an agent
MAX_STDERR = 5 * 1024 * 1024    # stderr kept per agent
TILE_THRESHOLDS = (1024, 2048, 4096, 8192, 16384)


@dataclass
class Config:
    image: str = "2048-eval:latest"
    use_docker: bool = True
    games: int = 100
    master_seed: int = 2048
    total_budget: float = 1800.0        # seconds: training + all games, per agent
    move_timeout: float = 5.0           # seconds per move (hang guard)
    first_move_timeout: float = 60.0    # first move of a process (JIT warm-up)
    illegal: str = "random"             # "random" or "forfeit"
    cpus: int = 2
    memory: str = "4g"
    pids: int = 256
    gpus: Optional[str] = None
    max_files: int = 2
    max_mb: float = 50.0
    mode: str = "sequential"            # "sequential" or "interleaved"


class SubmissionError(Exception):
    pass


class AgentFailure(Exception):
    def __init__(self, status, detail=""):
        super().__init__(status)
        self.status = status
        self.detail = detail


def make_seeds(master_seed, n):
    rng = random.Random(master_seed)
    return [rng.randrange(2 ** 32) for _ in range(n)]


def stage_submission(path, max_files=2, max_mb=50.0):
    """Validate a submission and copy it to a private staging dir.
    Returns (stage_dir, entry_filename, language, nickname)."""
    path = Path(path)
    if not path.exists():
        raise SubmissionError("path does not exist")
    if path.is_file():
        files = [path]
    else:
        items = sorted(path.iterdir())
        if any(p.is_dir() for p in items):
            raise SubmissionError("subdirectories are not allowed")
        files = [p for p in items if p.is_file()]
    entries = [f for f in files if NAME_RE.match(f.name)]
    if len(entries) != 1:
        raise SubmissionError("need exactly one file named agent_<name>.py or agent_<name>.jl "
                              f"(found {len(entries)})")
    counted = [f for f in files if f.name not in DEP_FILES]
    if len(counted) > max_files:
        raise SubmissionError(f"{len(counted)} files submitted, at most {max_files} allowed")
    if sum(f.stat().st_size for f in files) > max_mb * 2 ** 20:
        raise SubmissionError(f"submission larger than {max_mb:g} MB")
    m = NAME_RE.match(entries[0].name)
    stage = Path(tempfile.mkdtemp(prefix="sub_"))
    for f in files:
        shutil.copy2(f, stage / f.name)
        os.chmod(stage / f.name, 0o644)
    os.chmod(stage, 0o755)
    return stage, entries[0].name, ("python" if m.group(2) == "py" else "julia"), m.group(1)


class AgentRun:
    """One agent process plus its bookkeeping."""

    def __init__(self, cfg, path, outdir):
        self.cfg = cfg
        self.outdir = Path(outdir)
        self.stage, self.entry, self.lang, self.nick = stage_submission(path, cfg.max_files, cfg.max_mb)
        self.container = f"eval2048-{self.nick.lower()}-{uuid.uuid4().hex[:8]}"
        self.stderr_path = self.outdir / f"{self.nick}.stderr.log"
        self.games_path = self.outdir / f"{self.nick}.games.jsonl"
        self.status = "pending"   # pending/running/finished or a failure status
        self.detail = ""
        self.busy = 0.0
        self.train_seconds = 0.0
        self.games = []
        self.proc = None
        self.first_move = True
        self.q = queue.Queue()

    # ---- process management -------------------------------------------------
    @property
    def alive(self):
        return self.status == "running"

    def remaining(self):
        return self.cfg.total_budget - self.busy

    def _env_vars(self):
        n = str(self.cfg.cpus)
        return {
            "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
            "NUMEXPR_NUM_THREADS": n, "JULIA_NUM_THREADS": n,
            "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "HOME": "/tmp", "XDG_CACHE_HOME": "/tmp/.cache", "NUMBA_CACHE_DIR": "/tmp/numba",
            "MPLCONFIGDIR": "/tmp/mpl", "JULIA_DEPOT_PATH": "/tmp/julia_depot:/opt/julia_depot",
        }

    def _command(self):
        c = self.cfg
        if not c.use_docker:
            if self.lang == "python":
                return [sys.executable, "-B", str(HERE / "shim.py"), str(self.stage / self.entry)]
            return ["julia", f"--threads={c.cpus}", "--startup-file=no",
                    str(HERE / "shim.jl"), str(self.stage / self.entry)]
        cmd = ["docker", "run", "-i", "--rm", "--init", "--name", self.container,
               "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=2g",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--user", "65534:65534",
               "--memory", c.memory, "--memory-swap", c.memory,
               "--cpus", str(c.cpus), "--pids-limit", str(c.pids),
               "-v", f"{self.stage}:/sub:ro",
               "-v", f"{HERE / 'shim.py'}:/harness/shim.py:ro",
               "-v", f"{HERE / 'shim.jl'}:/harness/shim.jl:ro",
               "-w", "/tmp"]
        for k, v in self._env_vars().items():
            cmd += ["-e", f"{k}={v}"]
        if c.gpus:
            cmd += ["--gpus", c.gpus]
        cmd.append(c.image)
        if self.lang == "python":
            cmd += ["python", "-B", "/harness/shim.py", f"/sub/{self.entry}"]
        else:
            cmd += ["julia", "--project=/opt/julia_env", f"--threads={c.cpus}",
                    "--startup-file=no", "/harness/shim.jl", f"/sub/{self.entry}"]
        return cmd

    def _popen_env(self):
        if self.cfg.use_docker:
            return None
        skip = {"HOME", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "JULIA_DEPOT_PATH"}
        return {**os.environ, **{k: v for k, v in self._env_vars().items() if k not in skip}}

    def _read_stdout(self):
        try:
            while True:
                line = self.proc.stdout.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE and not line.endswith(b"\n"):
                    self.q.put(("overflow", None))
                    break
                self.q.put(("line", line))
        except (OSError, ValueError):
            pass
        finally:
            self.q.put(("eof", None))

    def _read_stderr(self):
        written = 0
        try:
            with open(self.stderr_path, "ab") as f:
                while True:
                    chunk = self.proc.stderr.read1(8192)
                    if not chunk:
                        break
                    if written < MAX_STDERR:
                        f.write(chunk[:MAX_STDERR - written])
                        f.flush()
                        written += len(chunk)
        except (OSError, ValueError):
            pass

    def _send(self, obj):
        try:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            self.proc.stdin.flush()
        except (OSError, ValueError):
            raise AgentFailure("crashed", "could not write to agent")

    def _recv(self, timeout, timeout_status):
        if timeout <= 0:
            raise AgentFailure("budget_exceeded")
        try:
            kind, line = self.q.get(timeout=timeout)
        except queue.Empty:
            raise AgentFailure(timeout_status)
        if kind == "eof":
            raise AgentFailure("crashed", "agent process exited")
        if kind == "overflow":
            raise AgentFailure("protocol_violation", "reply line too long")
        try:
            msg = json.loads(line)
        except ValueError:
            raise AgentFailure("protocol_violation", "reply was not JSON")
        if not isinstance(msg, dict):
            raise AgentFailure("protocol_violation", "reply was not a JSON object")
        return msg

    def _kill(self):
        if self.proc is None:
            return
        if self.cfg.use_docker:
            subprocess.run(["docker", "kill", self.container],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.proc.kill()
        except OSError:
            pass

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._kill()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        shutil.rmtree(self.stage, ignore_errors=True)

    # ---- lifecycle ------------------------------------------------------------
    def start(self, log=print):
        log(f"[{self.nick}] starting ({self.lang}) and training...")
        t0 = time.monotonic()
        try:
            self.proc = subprocess.Popen(self._command(), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         env=self._popen_env())
        except OSError as e:
            self.status, self.detail = "launch_failed", str(e)
            log(f"[{self.nick}] FAILED to launch: {e}")
            return False
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        try:
            msg = self._recv(self.remaining(), "budget_exceeded")
            if msg.get("ready") is True:
                self.status = "running"
            else:
                self.status = "train_failed"
                self.detail = str(msg.get("error", msg))[-800:]
        except AgentFailure as e:
            self.status = "train_failed" if e.status == "crashed" else e.status
            self.detail = e.detail
        self.train_seconds = time.monotonic() - t0
        self.busy += self.train_seconds
        if not self.alive:
            self._kill()
            log(f"[{self.nick}] {self.status} during training after {self.train_seconds:.1f}s {self.detail[:200]}")
            return False
        log(f"[{self.nick}] ready after {self.train_seconds:.1f}s")
        return True

    def play(self, idx, seed, log=print):
        t0 = time.monotonic()
        rng = random.Random(seed ^ 0xA5A5A5A5)   # fallback moves; independent of the env's RNG
        random.seed(seed)                        # the env draws tiles from the global RNG
        env = Game2048Env()
        state = env.reset()
        moves, illegal, errors, status = [], 0, 0, "ok"
        try:
            while not env.done:
                valid = env.get_available_actions()
                limit = self.cfg.first_move_timeout if self.first_move else self.cfg.move_timeout
                timeout_status = "timeout"
                left = self.remaining() - (time.monotonic() - t0)
                if left <= limit:
                    limit, timeout_status = left, "budget_exceeded"
                self._send({"state": state.tolist(), "valid": valid})
                msg = self._recv(limit, timeout_status)
                self.first_move = False
                action = msg.get("action")
                if "error" in msg:
                    errors += 1
                    action = None
                ok = isinstance(action, int) and not isinstance(action, bool) and action in valid
                if not ok:
                    if "error" not in msg:
                        illegal += 1
                    if self.cfg.illegal == "forfeit":
                        status = "illegal_move"
                        break
                    action = rng.choice(valid)
                moves.append(action)
                state, _, _, _ = env.step(action)
        except AgentFailure as e:
            status = e.status
            self.detail = e.detail
            self._kill()
        dt = time.monotonic() - t0
        self.busy += dt
        rec = {"game": idx, "seed": seed, "score": int(env.score), "max_tile": int(env.board.max()),
               "moves": len(moves), "illegal": illegal, "errors": errors, "status": status,
               "seconds": round(dt, 3), "actions": "".join(map(str, moves))}
        self.games.append(rec)
        with open(self.games_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"[{self.nick}] game {idx + 1}/{self.cfg.games} score={rec['score']} "
            f"max={rec['max_tile']} moves={rec['moves']} {dt:.1f}s"
            + ("" if status == "ok" else f"  ** {status}"))
        if status != "ok":
            self.status = status
        return rec

    def finish(self):
        if self.alive and len(self.games) == self.cfg.games:
            self.status = "finished"

    # ---- results --------------------------------------------------------------
    def summary(self):
        n = self.cfg.games
        scores = [g["score"] for g in self.games]
        padded = scores + [0] * (n - len(scores))      # unplayed games score 0
        tiles = [g["max_tile"] for g in self.games]
        sd = statistics.stdev(padded) if n > 1 else 0.0
        return {
            "nick": self.nick, "status": self.status, "detail": self.detail[:300],
            "games_played": len(self.games), "games_total": n,
            "mean_score": statistics.fmean(padded), "sem": sd / math.sqrt(n),
            "median_score": statistics.median(padded),
            "max_score": max(scores, default=0), "max_tile": max(tiles, default=0),
            "tile_rates": {str(t): sum(x >= t for x in tiles) / n for t in TILE_THRESHOLDS},
            "illegal_moves": sum(g["illegal"] for g in self.games),
            "agent_errors": sum(g["errors"] for g in self.games),
            "train_seconds": round(self.train_seconds, 1), "busy_seconds": round(self.busy, 1),
        }


def write_leaderboard(outdir, cfg, runs, rejected):
    rows = [r.summary() for r in runs]
    rows.sort(key=lambda s: (-s["mean_score"], -s["max_tile"]))
    data = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "games_per_agent": cfg.games,
            "leaderboard": rows, "rejected": rejected}
    tmp = Path(outdir) / "leaderboard.json.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, Path(outdir) / "leaderboard.json")      # atomic: safe for a live dashboard
    return rows


def print_table(rows, rejected):
    print(f"\n{'#':>2} {'agent':<16}{'mean':>10}{'+-sem':>8}{'median':>9}{'max tile':>10}"
          f"{'2048%':>7}{'games':>7}  status")
    for i, s in enumerate(rows, 1):
        print(f"{i:>2} {s['nick']:<16}{s['mean_score']:>10.0f}{s['sem']:>8.0f}{s['median_score']:>9.0f}"
              f"{s['max_tile']:>10}{100 * s['tile_rates']['2048']:>6.0f}%"
              f"{s['games_played']:>4}/{s['games_total']:<3} {s['status']}")
    for r in rejected:
        print(f" - REJECTED {r['submission']}: {r['detail']}")


def run_competition(cfg, submissions, outdir, log=print):
    """Run all submissions on this machine. Returns (runs, rejected)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    image_id = None
    if cfg.use_docker:
        p = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", cfg.image],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"Docker image '{cfg.image}' not found. Build it first:\n"
                             f"    docker build -t {cfg.image} .\n{p.stderr.strip()}")
        image_id = p.stdout.strip()
    else:
        log("WARNING: --no-docker means NO sandbox. Use only for debugging trusted code.")

    runs, rejected, seen = [], [], set()
    for path in submissions:
        try:
            r = AgentRun(cfg, path, outdir)
            if r.nick.lower() in seen:
                r.close()
                raise SubmissionError(f"duplicate nickname '{r.nick}'")
            seen.add(r.nick.lower())
            runs.append(r)
        except SubmissionError as e:
            rejected.append({"submission": str(path), "status": "rejected", "detail": str(e)})
            log(f"REJECTED {path}: {e}")

    seeds = make_seeds(cfg.master_seed, cfg.games)
    (outdir / "config.json").write_text(json.dumps({
        "config": asdict(cfg), "seeds": seeds, "image_id": image_id,
        "python": sys.version, "cpu_count": os.cpu_count(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "submissions": [r.nick for r in runs]}, indent=2))

    try:
        if cfg.mode == "interleaved":
            for r in runs:
                r.start(log)                      # train one at a time, alone on the machine
                write_leaderboard(outdir, cfg, runs, rejected)
            for gi, seed in enumerate(seeds):
                for r in runs:
                    if r.alive:
                        r.play(gi, seed, log)
                        write_leaderboard(outdir, cfg, runs, rejected)
            for r in runs:
                r.finish()
        else:
            for r in runs:
                if r.start(log):
                    for gi, seed in enumerate(seeds):
                        if not r.alive:
                            break
                        r.play(gi, seed, log)
                        write_leaderboard(outdir, cfg, runs, rejected)
                r.finish()
                r.close()
                write_leaderboard(outdir, cfg, runs, rejected)
    finally:
        for r in runs:
            r.close()
    rows = write_leaderboard(outdir, cfg, runs, rejected)
    print_table(rows, rejected)
    return runs, rejected


def main():
    d = Config()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("submissions", nargs="+", help="agent file(s) or directories")
    ap.add_argument("--out", default="results", help="results directory (a timestamped run dir is created)")
    ap.add_argument("--image", default=d.image)
    ap.add_argument("--no-docker", action="store_true", help="DEBUG ONLY: run agents unsandboxed")
    ap.add_argument("--games", type=int, default=d.games)
    ap.add_argument("--master-seed", type=int, default=d.master_seed)
    ap.add_argument("--total-budget", type=float, default=d.total_budget, help="seconds per agent")
    ap.add_argument("--move-timeout", type=float, default=d.move_timeout)
    ap.add_argument("--first-move-timeout", type=float, default=d.first_move_timeout)
    ap.add_argument("--illegal", choices=["random", "forfeit"], default=d.illegal)
    ap.add_argument("--cpus", type=int, default=d.cpus)
    ap.add_argument("--memory", default=d.memory)
    ap.add_argument("--pids", type=int, default=d.pids)
    ap.add_argument("--gpus", default=None, help='e.g. "all" (needs the NVIDIA container toolkit)')
    ap.add_argument("--max-files", type=int, default=d.max_files)
    ap.add_argument("--max-mb", type=float, default=d.max_mb)
    ap.add_argument("--mode", choices=["sequential", "interleaved"], default=d.mode)
    a = ap.parse_args()
    cfg = Config(image=a.image, use_docker=not a.no_docker, games=a.games, master_seed=a.master_seed,
                 total_budget=a.total_budget, move_timeout=a.move_timeout,
                 first_move_timeout=a.first_move_timeout, illegal=a.illegal, cpus=a.cpus,
                 memory=a.memory, pids=a.pids, gpus=a.gpus, max_files=a.max_files,
                 max_mb=a.max_mb, mode=a.mode)
    outdir = Path(a.out) / time.strftime("run_%Y%m%d_%H%M%S")
    print(f"Results in {outdir}")
    run_competition(cfg, a.submissions, outdir, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
