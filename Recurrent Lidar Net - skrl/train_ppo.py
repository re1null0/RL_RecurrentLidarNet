import os
import sys
import yaml
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_interface import make_vector_env
from utils import seed_everything, create_result_dir


def main():
    config_path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]).expanduser()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config.get("seed", 0))
    results_dir = create_result_dir(config["experiment_name"])
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    env = make_vector_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.get("learning_rate", 3e-4),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        n_steps=config.get("rollout_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("ppo_epochs", 10),
        clip_range=config.get("ppo_clip", 0.2),
        verbose=1,
        tensorboard_log=results_dir,
        device=device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("save_interval", 10000),
        save_path=results_dir,
        name_prefix="agent",
    )

    model.learn(total_timesteps=config["total_timesteps"], callback=checkpoint_callback)
    model.save(os.path.join(results_dir, "ppo_agent"))


if __name__ == "__main__":
    main()
