from omegaconf import DictConfig, OmegaConf
import torch
import argparse
import engine, utils
import gymnasium, model_builder
import numpy as np
import wandb
import os


def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_render = gymnasium.make('CartPole-v1', render_mode=None)
    if cfg.exercise_1.train:
        seed = cfg.exercise_1.train_seed
        utils.set_seed(seed, device)
        env = gymnasium.make('CartPole-v1')
        env.reset(seed=seed)
        env.action_space.seed(seed)
        policy = model_builder.policyNet(env).to(device)
        with wandb.init(project=cfg.project_name, name=cfg.run_name, config=OmegaConf.to_container(cfg,resolve=True)):
            engine.reinforce(cfg, env, policy, env_render, device)
        
    if cfg.exercise_1.eval:
        ckpt_path = os.path.join(cfg.checkpoint_dir, "cartpole_best.pt")
        policy = model_builder.policyNet(env_render)
        policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        (obs, acts, rews, log_acts) = engine.run_episode(cfg, env_render, policy)
        print(len(rews))
    
    if cfg.exercise_1.baseline:
        seed = 43
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = gymnasium.make('CartPole-v1')
        env.reset(seed=seed)
        policy = model_builder.policyNet(env).to(device)
        baseline = model_builder.baseline(env).to(device)
        with wandb.init(project=cfg.project_name, name=cfg.run_name, config=OmegaConf.to_container(cfg,resolve=True)):
            engine.reinforce_baseline(cfg, env, policy, baseline, env_render, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str,   default="DLA-Lab3/configs/config.yaml")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--run_name", type  = str, default = "CartPole")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.lr is not None: cfg.opt.lr = args.lr
    if args.run_name is not None: cfg.run_name = args.run_name
    main(cfg)