"""
shim.py - runs INSIDE the sandbox. Part of the trusted harness, not of a submission.

    python shim.py /sub/agent_<name>.py

Protocol (one JSON object per line):
    shim -> harness : {"ready": true}                    after train_and_instantiate()
    harness -> shim : {"state": [[...]], "valid": [..]}  one per move
    shim -> harness : {"action": 0..3}   or   {"error": "..."}

The real stdin/stdout are duplicated to private handles and fds 0/1 are
re-pointed (stdout -> stderr, stdin -> /dev/null), so stray print() calls or
C-library output from an agent cannot corrupt the protocol.
"""
import importlib.util
import json
import os
import sys
import traceback


def main():
    path = os.path.abspath(sys.argv[1])

    out = os.fdopen(os.dup(1), "w", buffering=1)
    inp = os.fdopen(os.dup(0), "r")
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(2, 1)
    os.dup2(devnull, 0)
    sys.stdout = sys.stderr
    sys.stdin = open(os.devnull)

    def send(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    import numpy as np

    sys.path.insert(0, os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location("agent_submission", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["agent_submission"] = mod
        spec.loader.exec_module(mod)
        for fn in ("train_and_instantiate", "act"):
            if not callable(getattr(mod, fn, None)):
                raise AttributeError(f"submission does not define {fn}()")
        agent = mod.train_and_instantiate()
    except BaseException:
        send({"error": traceback.format_exc()[-800:]})
        sys.exit(2)

    send({"ready": True})

    for line in inp:
        msg = json.loads(line)
        state = np.array(msg["state"], dtype=np.int64)
        state.setflags(write=False)
        valid = [int(v) for v in msg["valid"]]
        try:
            action = int(mod.act(agent, state, valid))
            send({"action": action})
        except Exception:
            send({"error": traceback.format_exc()[-800:]})


if __name__ == "__main__":
    main()
