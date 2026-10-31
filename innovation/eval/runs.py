"""
runs.py – one folder per experiment run under results/innovation/, with the
metadata needed to reproduce it (git commit, arguments, machine).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "innovation"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def machine() -> dict[str, Any]:
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    try:
        import torch
        info.update(torch=torch.__version__, torch_threads=torch.get_num_threads(), cuda=torch.cuda.is_available())
    except Exception:
        pass
    return info


def new_run(name: str, args: dict[str, Any] | None = None, root: Path = RESULTS_DIR) -> Path:
    """Create results/innovation/<YYYYmmdd_HHMMSS>_<name>/ with meta.json and return it."""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    out = root / run_id
    out.mkdir(parents=True, exist_ok=False)
    meta = {"run_id": run_id, "started": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv), "args": args or {}, "machine": machine()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
