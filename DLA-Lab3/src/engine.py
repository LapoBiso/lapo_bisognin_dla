from collections import deque
import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import wandb
import os

OPTIMIZERS = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}

def build_optimizer(params, cfg):
    name = cfg.opt.name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{cfg.opt.name}', expected one of {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](params, lr=cfg.opt.lr)


def compute_returns(rewards, gamma):
    returns = torch.zeros(len(rewards))
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def choose_action(cfg, obs, policy):
    logits = policy(obs)
    dist = Categorical(logits=logits / cfg.policy.temperature)
    action = dist.sample()
    log_prob = dist.log_prob(action).reshape(1)
    return action.item(), log_prob


def choose_action_deterministic(obs, policy):
    with torch.no_grad():
        logits = policy(obs)
        action = torch.argmax(logits).item()
    return action


def run_episode(cfg, env, policy, device, seed = None, deterministic = False):
    obs, acts, rews, log_acts = [], [], [], []
    if seed is not None:
        s, info = env.reset(seed = seed)
    else: 
        s, info = env.reset()
    for i in range(cfg.exercise_1.lenEp):
        s = torch.tensor(s, device = device, dtype=torch.float32)
        if deterministic:
            action = choose_action_deterministic(s, policy)
        else:
            action, log_action = choose_action(cfg, s, policy)
            log_acts.append(log_action)
        obs.append(s)
        acts.append(action)
        s, reward, term, trunc, info = env.step(action)
        rews.append(reward)
        if term or trunc:
            break
    log_out = torch.cat(log_acts) if log_acts else None
    return torch.stack(obs), acts, rews, log_out

def evaluate_reinforce(cfg, env_render, policy, episode, device):
    if env_render:
        policy.eval()
        with torch.no_grad():
            eval_returns, eval_lengths = [], []
            for eval_episode in range(cfg.exercise_1.M):
                seed = cfg.exercise_1.eval_seed + eval_episode
                obs, acts, rews, log_acts = run_episode(cfg, env_render, policy, device, seed = seed, deterministic = False)
                eval_returns.append(sum(rews))
                eval_lengths.append(len(rews))
            avg_eval_returns = np.mean(eval_returns)
            avg_eval_lengths = np.mean(eval_lengths)
            wandb.log({
                "eval/avg_return": avg_eval_returns,
                "eval/avg_length": avg_eval_lengths,
            }, step=episode)
            


def reinforce(cfg, env, policy, env_render, device):
    opt = build_optimizer(policy.parameters(), cfg)
    run_rews = [0.0]
    best_rew = 0.0
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_best.pt")
    for episode in range(cfg.exercise_1.maxIt):
        (obs, acts, rews, log_acts) = run_episode(cfg, env, policy, device)
        returns = torch.tensor(compute_returns(rews, cfg.policy.gamma), dtype = torch.float32, device = device)
        tot_rew = sum(rews)
        run_rews.append(0.05 * tot_rew + 0.95 * run_rews[-1])
        
        if episode > cfg.exercise_1.warmup and run_rews[-1] > best_rew + cfg.exercise_1.min_delta:
            best_rew = run_rews[-1]
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            torch.save(policy.state_dict(), ckpt_path)
            wandb.log({"checkpoint/best_reward": best_rew}, step=episode)

        if len(returns) > 2:
            returns = (returns - returns.mean()) / returns.std()
        
        opt.zero_grad()
        loss = -(log_acts * returns).mean()
        loss.backward()
        opt.step()

        if not episode % cfg.exercise_1.N:
            evaluate_reinforce(cfg, env_render, policy, episode, device)
            policy.train()
        wandb.log({"train/episode": episode, "train/running_reward":run_rews[-1]}, step = episode)

    return run_rews


def reinforce_baseline(cfg, env, policy, baseline, env_render, device):
    opt_policy = build_optimizer(policy.parameters(), cfg)
    opt_baseline = build_optimizer(baseline.parameters(), cfg)
    run_rews = [0.0]
    best_rew = 0.0
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_baseline.pt")
    for episode in range(cfg.exercise_1.maxIt):
        (obs, acts, rews, log_acts) = run_episode(cfg, env, policy, device)
        returns = torch.tensor(compute_returns(rews, cfg.policy.gamma), dtype = torch.float32, device = device)
        tot_rew = sum(rews)
        run_rews.append(0.05 * tot_rew + 0.95 * run_rews[-1])

        if episode > cfg.exercise_1.warmup and run_rews[-1] > best_rew + cfg.exercise_1.min_delta:
            best_rew = run_rews[-1]
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            torch.save(policy.state_dict(), ckpt_path)
            wandb.log({"checkpoint/best_reward": best_rew}, step=episode)
        
        values = baseline(obs)
        advantage = returns - values.detach()
        advantage = (advantage - advantage.mean()) / advantage.std()

        opt_policy.zero_grad()
        policy_loss = -(log_acts * advantage).mean()
        policy_loss.backward()
        opt_policy.step()

        opt_baseline.zero_grad() 
        baseline_loss = F.mse_loss(values, returns)
        baseline_loss.backward()
        opt_baseline.step()
        
        if not episode % cfg.exercise_1.N:
            evaluate_reinforce(cfg, env_render, policy, episode, device)
            policy.train()
        wandb.log({"train/episode": episode, "train/running_reward":run_rews[-1]}, step = episode)


def choose_deepQ_action(env, state, q_net, device, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype = torch.uint8, device = device).unsqueeze(0).float() / 255.0
            q_values = q_net(state_t)
            return torch.argmax(q_values, dim = 1).item()
        
def linear_epsilon(cfg, global_step):
    frac = min(global_step / cfg.exercise_3.eps_decay_step, 1.0)
    return cfg.exercise_3.epsilon + frac * (cfg.exercise_3.eps_min - cfg.exercise_3.epsilon) 


def deepQ_learning(cfg, env, q_net, t_net, replay, device, eval_seeds):
    opt = build_optimizer(q_net.parameters(), cfg)
    epsilon = cfg.exercise_3.epsilon
    global_step = 0
    best_mean_reward = -float("inf")
    rews = [0.0]
    ckpt_dir = cfg.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    for episode in range(cfg.exercise_3.maxIt):
        (state, info) = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        for t in range(cfg.exercise_3.lenEp):
            action = choose_deepQ_action(env, state, q_net, device, epsilon)
            (next_state, reward, term, trunc, info) = env.step(action)
            done = term or trunc
            replay.append(state, action, reward, next_state, term)
            state = next_state
            global_step +=1
            ep_reward += reward
            ep_steps += 1

            if len(replay) >= cfg.exercise_3.warmup and global_step % 4 == 0:
                states, actions, rewards, next_states, dones = replay.sample(cfg.exercise_3.batch_size)
                states      = states.to(device, non_blocking=True).float() / 255.0
                actions     = actions.to(device, non_blocking=True)
                rewards     = rewards.to(device, non_blocking=True)
                next_states = next_states.to(device, non_blocking=True).float() / 255.0
                dones       = dones.to(device, non_blocking=True)
                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = t_net(next_states).max(dim=1).values
                    targets = rewards + cfg.policy.gamma * max_next_q * (1 - dones)

                loss = F.smooth_l1_loss(q_values, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()

                wandb.log({
                    "train/loss": loss.item(),
                    "train/mean_q": q_values.mean().item(),
                }, step=global_step)

                if global_step % cfg.exercise_3.target_update == 0:
                    t_net.load_state_dict(q_net.state_dict())

            if done:
                break
        epsilon = linear_epsilon(cfg, global_step)

        rews.append(0.05 * ep_reward + 0.95 * rews[-1])
        mean_reward = rews[-1]

        wandb.log({
            "rollout/ep_length": ep_steps,
            "rollout/epsilon": epsilon,
            "rollout/mean_reward": mean_reward,
            "episode": episode,
        }, step=global_step)

        if len(rews) >= 50 and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            ckpt_path = os.path.join(ckpt_dir, f"{cfg.run_name}.pt")
            torch.save({"model_state_dict": q_net.state_dict(),}, ckpt_path)

        if episode and episode % cfg.exercise_3.N == 0:
            evaluate_DeepQ(cfg, env, q_net, device, eval_seeds, global_step=global_step)

def evaluate_DeepQ(cfg, env, q_net, device, eval_seeds, global_step=None):
    q_net.eval()
    eval_returns = []
    eval_lengths = []
    frames = []
    with torch.no_grad():
        for seed in eval_seeds:
            (state, info) = env.reset(seed = seed)
            ep_reward = 0.0
            ep_steps = 0
            for t in range(cfg.exercise_3.lenEp):
                action = choose_deepQ_action(env, state, q_net, device, epsilon=0.0)
                (state, reward, term, trunc, info) = env.step(action)
                ep_reward += reward
                ep_steps += 1
                if term or trunc:
                    break
            eval_returns.append(ep_reward)
            eval_lengths.append(ep_steps)
    q_net.train()

    avg_eval_return = np.mean(eval_returns)
    avg_eval_length = np.mean(eval_lengths)
    wandb.log({"eval/avg_return": avg_eval_return, "eval/avg_length": avg_eval_length}, step=global_step)
    return avg_eval_return


def play_episode_DeepQ(cfg, env, q_net, device):
    q_net.eval()
    ep_reward = 0.0
    ep_steps = 0
    with torch.no_grad():
        (state, info) = env.reset()
        for t in range(cfg.exercise_3.lenEp):
            action = choose_deepQ_action(env, state, q_net, device, epsilon=0.0)
            (state, reward, term, trunc, info) = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if term or trunc:
                break
    q_net.train()
    return ep_reward, ep_steps

