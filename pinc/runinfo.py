"""Run bookkeeping: run directories, git hash, package versions, MANIFEST (ground rules 4, 6)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys

from .config import Config, RESULTS_DIR, ROOT


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def versions() -> dict:
    import numpy, scipy, tensorflow, matplotlib, yaml  # noqa
    return dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
                tensorflow=tensorflow.__version__, matplotlib=matplotlib.__version__, pyyaml=yaml.__version__)


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def command_line() -> str:
    return "python " + " ".join(sys.argv)


def make_run_dir(exp: str, run_id: str) -> str:
    d = os.path.join(RESULTS_DIR, exp, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_meta(run_dir: str, cfg: Config, seed: int, extra: dict | None = None):
    cfg.to_json(os.path.join(run_dir, "config.json"))
    meta = dict(seed=seed, git=git_hash(), versions=versions(), cpu=cpu_model(),
                platform=platform.platform(), command=command_line(),
                timestamp=_dt.datetime.now().isoformat(timespec="seconds"))
    if extra:
        meta.update(extra)
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return meta


def manifest_append(artifacts, command: str | None = None):
    """Append `artifact -> command -> git hash` lines to results/MANIFEST.md."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "MANIFEST.md")
    new = not os.path.exists(path)
    command = command or command_line()
    h = git_hash()
    if isinstance(artifacts, str):
        artifacts = [artifacts]
    with open(path, "a") as fh:
        if new:
            fh.write("# Results manifest\n\nartifact | command | git hash\n---|---|---\n")
        for a in artifacts:
            rel = os.path.relpath(a, ROOT)
            fh.write(f"`{rel}` | `{command}` | `{h}`\n")


def save_json(obj, path):
    def conv(o):
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=conv)
