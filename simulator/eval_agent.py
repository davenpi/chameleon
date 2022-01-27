import argparse
import numpy as np
import chameleon as ch
from stable_baselines3 import PPO


parser = argparse.ArgumentParser()

parser.add_argument("model", type=str, help="Path to model.")
parser.add_argument(
    "-a", "--atol", default=0.007, type=float, help="Absolute error tolerance."
)
parser.add_argument(
    "-t", "--first_target", default=0.19, type=float, help="First target to reach"
)
parser.add_argument(
    "-n", "--nevals", default=5, type=int, help="Number of evaluation episodes to run"
)

args = parser.parse_args()
model_path = args.model
atol = args.atol
n = args.nevals
target = args.first_target

model = PPO.load(model_path)

targets = [target]
if n > 0:
    targets += list((0.2 - 0.1) * np.random.sample(n) + 0.1)

rewards = []
for tar in targets:
    env = ch.Chameleon(
        target_pos=tar, atol=atol, train=False
    )  # assuming default parameter values in chameleon for now
    ep_rew = 0
    state = env.reset()
    done = False
    while not done:
        action = model.predict(state, deterministic=True)[0]
        print(action)
        state, rew, done, _ = env.step(action)
        ep_rew += rew
    rewards.append(ep_rew)
    print(f"Target: {tar}")
    print(f"Score: {ep_rew}")
