from omegaconf import DictConfig, OmegaConf
import torch
import argparse
import engine, utils
import gymnasium, model_builder
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
import os


from stable_baselines3 import PPO
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation, ResizeObservation

def make_env():
    return gymnasium.make('CarRacing-v3', continuous=False, render_mode='rgb_array')

def make_eval_env():
    return gymnasium.make('CarRacing-v3', continuous=False, render_mode='human')

def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_env()
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)
    print(env.observation_space.shape)

    if cfg.exercise_3.model == "deepq" and cfg.exercise_3.train:
        eval_seeds = [1000 + i for i in range(cfg.exercise_3.M)]
        with wandb.init(project=cfg.project_name, name=cfg.run_name, config=OmegaConf.to_container(cfg,resolve=True)):
            q_net = model_builder.DeepQAgent(env).to(device)
            t_net = model_builder.DeepQAgent(env).to(device)
            replay = utils.replayMemory(cfg.exercise_3.capacity)
            engine.deepQ_learning(cfg,env,q_net, t_net, replay, device, eval_seeds)
    
    if cfg.exercise_3.eval:
        eval_env = make_eval_env()
        eval_env = GrayscaleObservation(eval_env)
        eval_env = FrameStackObservation(eval_env, stack_size=4)
        q_net = model_builder.DeepQAgent(eval_env).to(device)
        utils.load_model(q_net, 'DLA-Lab3/checkpoints/CarRacing_500eps_50kDecay_tu1000_0_995gamma.pt')
        ep_reward, ep_steps = engine.play_episode_DeepQ(cfg, eval_env, q_net, device)
        print(f"Episode reward: {ep_reward}, length: {ep_steps}")
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str,   default="DLA-Lab3/configs/config.yaml")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--run_name", type  = str, default = "CarRacing")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.lr is not None: cfg.opt.lr = args.lr
    if args.run_name is not None: cfg.run_name = args.run_name
    main(cfg)