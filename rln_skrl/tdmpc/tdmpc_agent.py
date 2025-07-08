import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PlanningConfig:
    horizon: int = 5
    iterations: int = 5
    num_samples: int = 512
    num_elites: int = 64
    gamma: float = 0.99


class LatentDynamicsModel(nn.Module):
    """Simple latent dynamics with reward, value and policy heads."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.dynamics = nn.GRUCell(action_dim, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.value_head = nn.Linear(latent_dim, 1)
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, action_dim),
        )
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(action, latent)

    def reward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.reward_head(latent)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        return self.value_head(latent)

    def policy(self, latent: torch.Tensor) -> torch.Tensor:
        return self.policy_head(latent)


class TDMPCAgent:
    def __init__(self, env, config: dict, device: torch.device, results_dir: str):
        self.env = env
        self.cfg = config
        self.device = device
        obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, "shape"):
            self.action_dim = int(np.prod(env.action_space.shape))
        else:
            self.action_dim = env.action_space.n
        latent_dim = config.get("latent_dim", 64)
        self.model = LatentDynamicsModel(obs_dim, self.action_dim, latent_dim).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.get("learning_rate", 3e-4))
        self.buffer = deque(maxlen=config.get("replay_buffer_size", 100000))
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.plan_cfg = PlanningConfig(
            horizon=config.get("planning_horizon", 5),
            iterations=config.get("planning_iterations", 5),
            num_samples=config.get("planning_samples", 512),
            num_elites=config.get("planning_elites", 64),
            gamma=config.get("gamma", 0.99),
        )
        self.results_dir = results_dir

    def plan_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z0 = self.model.encode(obs_t)
        mean = torch.zeros(self.plan_cfg.horizon, self.action_dim, device=self.device)
        std = torch.ones_like(mean)
        for _ in range(self.plan_cfg.iterations):
            actions = torch.randn(self.plan_cfg.num_samples, self.plan_cfg.horizon, self.action_dim, device=self.device) * std + mean
            values = torch.zeros(self.plan_cfg.num_samples, device=self.device)
            z = z0.repeat(self.plan_cfg.num_samples, 1)
            for t in range(self.plan_cfg.horizon):
                a = actions[:, t]
                z = self.model.next_latent(z, a)
                r = self.model.reward(z).squeeze(-1)
                values += (self.plan_cfg.gamma ** t) * r
            values += (self.plan_cfg.gamma ** self.plan_cfg.horizon) * self.model.value(z).squeeze(-1)
            elite_idx = torch.topk(values, self.plan_cfg.num_elites, largest=True).indices
            elite_actions = actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-4
        return mean[0].cpu().numpy()

    def update(self):
        batch = random.sample(self.buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        latent = self.model.encode(obs)
        next_latent = self.model.encode(next_obs)
        pred_next = self.model.next_latent(latent, actions)
        pred_reward = self.model.reward(pred_next)
        value = self.model.value(latent)
        next_value = self.model.value(next_latent)
        target = rewards + self.gamma * (1 - dones) * next_value.detach()

        value_loss = F.mse_loss(value, target)
        reward_loss = F.mse_loss(pred_reward, rewards)
        dyn_loss = F.mse_loss(pred_next, next_latent.detach())
        loss = value_loss + reward_loss + dyn_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        obs, _ = self.env.reset()
        episode_reward = 0.0
        for step in range(self.cfg.get("total_timesteps", 100000)):
            action = self.plan_action(obs)
            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = bool(term[0] or trunc[0])
            self.buffer.append((obs, action, float(reward[0]), next_obs[0], done))
            obs = self.env.reset()[0] if done else next_obs[0]
            episode_reward += reward[0]
            if done:
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0.0
            if len(self.buffer) >= self.batch_size:
                self.update()
        torch.save(self.model.state_dict(), f"{self.results_dir}/tdmpc_final_model.pth")
