import os
import sys
import yaml
from pathlib import Path

import torch
from stable_baselines3 import DQN
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

    exploration_fraction = 0.0
    if config.get("epsilon_decay", 0) > 0:
        exploration_fraction = config["epsilon_decay"] / float(config["total_timesteps"])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config.get("learning_rate", 1e-4),
        buffer_size=config.get("replay_buffer_size", 10000),
        batch_size=config.get("batch_size", 64),
        gamma=config.get("gamma", 0.99),
        target_update_interval=config.get("target_update_interval", 1000),
        exploration_initial_eps=config.get("epsilon_start", 1.0),
        exploration_final_eps=config.get("epsilon_end", 0.05),
        exploration_fraction=exploration_fraction,
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
    model.save(os.path.join(results_dir, "dqn_agent"))


if __name__ == "__main__":
    main()
