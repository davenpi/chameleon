import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from chameleon import Chameleon
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-tt",
    "--total_timesteps",
    type=int,
    help="Number to multiply by 2048 to get total number of timesteps",  # deals with n_steps=2048 causing miscounting
    default=int(2.5e2),
)
parser.add_argument(
    "-tp",
    "--target_position",
    type=float,
    help="Initial target position to reach to.",
    default=0.18,
)
parser.add_argument("-E", type=float, help="Young's Modulus of tongue.", default=1.0)
parser.add_argument(
    "-m",
    "--monitor",
    type=bool,
    help="Monitor the model during training.",
    default=True,
)
parser.add_argument(
    "-a", "--atol", type=float, help="Allowed error in reaching.", default=0.05
)
parser.add_argument(
    "-i", "--iterations", type=int, help="Number of iterations of training.", default=1
)

args = parser.parse_args()
target_pos = args.target_position
timesteps = int(2048 * args.total_timesteps)
monitor = args.monitor
atol = args.atol
E = args.E
its = args.iterations

print(f"Time steps: " + "{:.3e}".format(timesteps))
print(f"Iterations: {its}")
print(f"Allowed error: {atol}")


# location to save monitors
monitor_dir = f"monitors_a{atol}_target{target_pos}"
os.makedirs(monitor_dir, exist_ok=True)

# location to save trained agents
agents_dir = f"agents_a{atol}_target{target_pos}"
os.makedirs(agents_dir, exist_ok=True)

env = Chameleon(E=E, target_pos=target_pos, atol=atol)
eval_env = Chameleon(E=E, target_pos=target_pos, atol=atol)


def load_plot_results(monitor_file: str, monitor_dir: str) -> None:
    rews = np.loadtxt(monitor_file, delimiter=",", usecols=0, skiprows=2)
    num_eps = rews.shape[0]
    keep_after = 100 * math.floor(
        int(num_eps / 100)
    )  # just want number of episodes kept to be divisible by 100
    rews = rews[:keep_after:]
    rews = rews.reshape((-1, 100))
    means = rews.mean(axis=1)
    # Plot and save results
    plt.plot(means)
    plt.title("Episode reward over time")
    plt.xlabel("Epoch (100 episodes)")
    plt.ylabel("Mean reward")
    plt.savefig(monitor_dir + f"/rew_plot{i}.png")
    plt.clf()


callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-0.1, verbose=0)
for i in range(its):
    monitor_file = monitor_dir + f"/run{i}.monitor.csv"
    eval_file = monitor_dir + f"/run{i}.eval.csv"
    best_agent_path = agents_dir + f"/run{i}"
    if monitor:
        env = Monitor(env, filename=monitor_file)
        eval_env = Monitor(eval_env, filename=eval_file)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_agent_path,
        callback_on_new_best=callback_on_best,
        eval_freq=int(timesteps / 10),
        deterministic=True,
        render=False,
        verbose=0,
    )

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    # Training results
    load_plot_results(monitor_file, monitor_dir)
