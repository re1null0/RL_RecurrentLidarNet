# RL Recurrent Lidar Net

This project demonstrates training a recurrent policy for the F1TENTH simulator using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

The original version relied on the **skrl** library.  All training scripts have been rewritten to use Stable-Baselines3.

Training entry points are:

- `Recurrent Lidar Net - skrl/train_ppo.py`
- `Recurrent Lidar Net - skrl/train_dqn.py`

A configuration file can be provided as the first argument; otherwise `configs/default.yaml` is used.
