"""Evaluate DeepQ CarRacing checkpoints over a fixed set of episodes/seeds.

For each checkpoint, runs `--num_episodes` evaluation episodes (deterministic
policy, epsilon=0) using the same fixed seeds every time, then logs a single
score per checkpoint to wandb: the reward mean over all episodes.

Usage (from repo root):
    python DLA-Lab3/scripts/eval_checkpoints.py
"""
import argparse
import sys
from pathlib import Path

import gymnasium
import numpy as np
import torch
import wandb
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "DLA-Lab3" / "src"))

import engine, model_builder  # noqa: E402

CHECKPOINTS = {
    "CarRacing_tu2000": "CarRacing_500eps_50kDecay.pt",
    "CarRacing_tu1000_gamma0_995": "CarRacing_500eps_50kDecay_tu1000_0_995gamma.pt"
}


def make_env():
    env = gymnasium.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)
    return env


def evaluate_checkpoint(cfg, ckpt_path, seeds, device):
    env = make_env()
    q_net = model_builder.DeepQAgent(env).to(device)
    engine.load_deepQ_model(q_net, ckpt_path, device)
    q_net.eval()

    rewards = []
    with torch.no_grad():
        for seed in seeds:
            state, info = env.reset(seed=seed)
            ep_reward = 0.0
            for _ in range(cfg.exercise_3.lenEp):
                action = engine.choose_deepQ_action(env, state, q_net, device, epsilon=0.0)
                state, reward, term, trunc, info = env.step(action)
                ep_reward += reward
                if term or trunc:
                    break
            rewards.append(ep_reward)

    env.close()
    return rewards


def main(cfg, num_episodes, project):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(num_episodes))

    for run_name, ckpt_file in CHECKPOINTS.items():
        ckpt_path = REPO_ROOT / cfg.checkpoint_dir / ckpt_file
        print(f"\n=== Evaluating {run_name} ({ckpt_path}) over {num_episodes} episodes ===")

        rewards = evaluate_checkpoint(cfg, str(ckpt_path), seeds, device)
        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards))
        print(f"{run_name}: reward_mean={reward_mean:.2f} reward_std={reward_std:.2f}")

        with wandb.init(
            project=project,
            name=f"eval_{run_name}",
            config={"checkpoint": ckpt_file, "num_episodes": num_episodes, "seeds": seeds},
        ):
            wandb.log({"eval/reward_mean": reward_mean, "eval/reward_std": reward_std})

    print("\nAll checkpoint evaluations completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="DLA-Lab3/configs/config.yaml")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--project", type=str, default="carracing-eval")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, args.num_episodes, args.project)
