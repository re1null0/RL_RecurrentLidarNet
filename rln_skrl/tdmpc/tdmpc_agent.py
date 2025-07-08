import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

class TransitionModel(nn.Module):
    """Simple deterministic transition model predicting next state and reward."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.net(x)
        next_state = self.state_head(h)
        reward = self.reward_head(h)
        return next_state, reward

class ValueModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

class PolicyModel(nn.Module):
    def __init__(self, obs_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.low = torch.as_tensor(action_low, dtype=torch.float32)
        self.high = torch.as_tensor(action_high, dtype=torch.float32)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        a = self.net(state)
        return self.low + (a + 1) * 0.5 * (self.high - self.low)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, act, rew, next_obs, done):
        self.buffer.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, act, rew, nxt, done = zip(*(self.buffer[i] for i in indices))
        return (
            torch.as_tensor(np.array(obs), dtype=torch.float32),
            torch.as_tensor(np.array(act), dtype=torch.float32),
            torch.as_tensor(np.array(rew), dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(nxt), dtype=torch.float32),
            torch.as_tensor(np.array(done), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

class TDMPCAgent:
    def __init__(self, env, config, device=None):
        self.env = env
        self.cfg = config
        self.device = device or torch.device('cpu')

        obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'n'):
            raise ValueError('TD-MPC requires continuous action space')
        action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.model = TransitionModel(obs_dim, action_dim).to(self.device)
        self.value = ValueModel(obs_dim).to(self.device)
        self.policy = PolicyModel(obs_dim, action_dim, self.action_low, self.action_high).to(self.device)

        lr = config.get('learning_rate', 3e-4)
        self.optim_model = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optim_value = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
        self.batch_size = config.get('batch_size', 256)
        self.gamma = config.get('gamma', 0.99)
        self.horizon = config.get('horizon', 5)
        self.planning_iters = config.get('planning_iters', 100)
        self.random_steps = config.get('random_steps', 1000)

    def plan(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        best_return = -float('inf')
        best_action = None
        for _ in range(self.planning_iters):
            total_reward = 0.0
            s = state.clone()
            actions = []
            for t in range(self.horizon):
                a = torch.rand(1, self.env.action_space.shape[0], device=self.device)
                a = self.action_low + a * (self.action_high - self.action_low)
                actions.append(a)
                with torch.no_grad():
                    s, r = self.model(s, a)
                total_reward += r.item()
            with torch.no_grad():
                total_reward += self.value(s).item()
            if total_reward > best_return:
                best_return = total_reward
                best_action = actions[0]
        return best_action.squeeze(0).cpu().numpy()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        obs, act, rew, nxt, done = self.buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        act = act.to(self.device)
        rew = rew.to(self.device)
        nxt = nxt.to(self.device)
        done = done.to(self.device)

        # model loss
        pred_nxt, pred_rew = self.model(obs, act)
        loss_model = F.mse_loss(pred_nxt, nxt) + F.mse_loss(pred_rew, rew)
        self.optim_model.zero_grad()
        loss_model.backward()
        self.optim_model.step()

        # value loss
        with torch.no_grad():
            target = rew + self.gamma * (1 - done) * self.value(nxt)
        val = self.value(obs)
        loss_val = F.mse_loss(val, target)
        self.optim_value.zero_grad()
        loss_val.backward()
        self.optim_value.step()

        # policy loss (gradient through model and value)
        self.optim_policy.zero_grad()
        a = self.policy(obs)
        nxt_pred, r_pred = self.model(obs, a)
        loss_pi = -self.value(nxt_pred).mean()
        loss_pi.backward()
        self.optim_policy.step()

    def train(self):
        obs, _ = self.env.reset()
        for t in range(self.cfg['total_timesteps']):
            if t < self.random_steps:
                action = self.env.action_space.sample()
            else:
                action = self.plan(obs)
            next_obs, reward, done, truncated, _ = self.env.step(action)
            done_flag = done or truncated
            self.buffer.add(obs, action, reward, next_obs, done_flag)
            obs = next_obs if not done_flag else self.env.reset()[0]
            self.update()
            if (t + 1) % self.cfg.get('save_interval', 10000) == 0:
                self.save(self.cfg.get('results_dir', './'))

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'value': self.value.state_dict(),
            'policy': self.policy.state_dict()
        }, os.path.join(out_dir, 'tdmpc_model.pth'))

