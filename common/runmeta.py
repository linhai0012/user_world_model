"""Run config + run-meta I/O (CONVENTIONS.md §1, §3).

Loads `experiments/configs/<run_id>.yaml`, expands `${UWM_*}` env vars, and writes
`run_meta.json` into the run dir on scratch.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

_VAR = re.compile(r"\$\{(\w+)\}")


def _expand(obj):
    if isinstance(obj, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(run_id: str) -> dict:
    path = repo_root() / "experiments" / "configs" / f"{run_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config {path} (copy experiments/configs/_TEMPLATE.yaml)")
    cfg = _expand(yaml.safe_load(path.read_text()))
    if cfg.get("run_id") != run_id:
        raise ValueError(f"run_id mismatch: file says {cfg.get('run_id')!r}, asked {run_id!r}")
    return cfg


def run_dir(run_id: str) -> Path:
    base = os.environ.get("UWM_RUNS")
    if not base:
        raise RuntimeError("UWM_RUNS unset — `source scripts/env.sh` first.")
    d = Path(base) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_meta(run_id: str, cfg: dict, extra: dict | None = None) -> Path:
    meta = {
        "run_id": run_id,
        "config": cfg,
        "host": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    p = run_dir(run_id) / "run_meta.json"
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return p
