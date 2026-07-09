"""Run a sequence of DeepQ (exercise_3) experiments, config otherwise unchanged.

Experiments (run in order):
    1. opt.lr = 5e-5
    2. opt.lr = 1e-3

Usage (from repo root):
    python DLA-Lab3/scripts/run_experiments.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "DLA-Lab3" / "configs" / "config.yaml"
EXERCISE_3_SCRIPT = REPO_ROOT / "DLA-Lab3" / "src" / "exercise_3.py"

EXPERIMENTS = [
    {"run_name": "CarRacing_lr5e-5", "overrides": {"opt.lr": 5e-5}},
    {"run_name": "CarRacing_lr1e-3", "overrides": {"opt.lr": 1e-3}},
]


def run_experiment(run_name: str, overrides: dict, tmp_dir: Path) -> None:
    cfg = OmegaConf.load(BASE_CONFIG)
    for dotted_key, value in overrides.items():
        OmegaConf.update(cfg, dotted_key, value, merge=True)

    tmp_config_path = tmp_dir / f"{run_name}.yaml"
    OmegaConf.save(cfg, tmp_config_path)

    print(f"\n=== Running experiment: {run_name} (overrides: {overrides}) ===")
    result = subprocess.run(
        [
            sys.executable,
            str(EXERCISE_3_SCRIPT),
            "--config", str(tmp_config_path),
            "--run_name", run_name,
        ],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Experiment '{run_name}' failed with exit code {result.returncode}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dla_lab3_experiments_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for experiment in EXPERIMENTS:
            run_experiment(experiment["run_name"], experiment["overrides"], tmp_dir_path)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
