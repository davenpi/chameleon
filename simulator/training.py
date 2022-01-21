import argparse
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
    StopTrainingOnRewardThreshold,
)
from chameleon import Chameleon

parser = argparse.ArgumentParser()
parser.add_argument(
    "-tt",
    "--total_timesteps",
    type=int,
    help="Total timesteps argument for algorithm to learn on",
    default=int(6e5),
)
parser.add_argument(
    "-tp",
    "--target_position",
    type=float,
    help="Target position to reach to",
    default=0.18,
)
parser.add_argument("-E", type=float, help="Young's Modulus of tongue", default=1.0)
parser.add_argument(
    "-s", "--save", type=bool, help="Save model after training", default=True
)
parser.add_argument(
    "-m",
    "--monitor",
    type=bool,
    help="Monitor the model during training",
    default=True,
)
parser.add_argument(
    "-r", "--rtol", type=float, help="Allowed error in reaching", default=0.05
)
args = parser.parse_args()
target_pos = args.target_position
timesteps = args.total_timesteps
monitor = args.monitor
rtol = args.rtol
E = args.E

env = Chameleon(E=E, target_pos=target_pos, rtol=rtol)
eval_env = Chameleon(E=E, target_pos=target_pos, rtol=rtol)

if monitor:
    env = Monitor(env, filename=f"logs/r{rtol}target{target_pos}time{timesteps}")
    eval_env = Monitor(eval_env, filename="eval_logger")


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=50_000,
    deterministic=True,
    render=False,
    verbose=1,
)
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
)

model.learn(total_timesteps=timesteps, callback=eval_callback)

if args.save:
    model.save(f"ppo_r{rtol}target{target_pos}time{timesteps}")
