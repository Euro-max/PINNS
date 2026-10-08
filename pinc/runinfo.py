"""Run bookkeeping: run directories, git hash, package versions, MANIFEST (ground rules 4, 6)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys

from .config import Config, RESULTS_DIR, ROOT


CODE_PATHS = ("pinc", "experiments", "configs", "scripts")   # what a run's result depends on (results/ is not code)


def code_diff() -> str:
    """Uncommitted changes (staged or not) to the code paths, as a patch; empty when they are clean."""
    try:
        return subprocess.check_output(["git", "diff", "HEAD", "--", *CODE_PATHS], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode(errors="replace")
    except Exception:
        return ""


def git_hash() -> str:
    """HEAD, with "-dirty" when the code paths differ from it (changes to results/ do not count)."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "HEAD", "--quiet", "--", *CODE_PATHS], cwd=ROOT) != 0
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


def relative(arg: str) -> str:
    """An argument with the repository prefix removed, so recorded commands do not depend on the machine."""
    root = ROOT.rstrip(os.sep) + os.sep
    return arg.replace(root, "") if root in arg else arg


def command_line() -> str:
    return "python " + " ".join(relative(a) for a in sys.argv)


def make_run_dir(exp: str, run_id: str) -> str:
    d = os.path.join(RESULTS_DIR, exp, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_meta(run_dir: str, cfg: Config, seed: int, extra: dict | None = None):
    cfg.to_json(os.path.join(run_dir, "config.json"))
    h = git_hash()
    meta = dict(seed=seed, git=h, versions=versions(), cpu=cpu_model(),
                platform=platform.platform(), command=command_line(),
                timestamp=_dt.datetime.now().isoformat(timespec="seconds"))
    if h.endswith("-dirty"):
        meta["code_diff"] = code_diff()            # the uncommitted code the run used
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
