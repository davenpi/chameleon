import pickle
import argparse
import numpy as np
import chameleon as ch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()

parser.add_argument("model", type=str, help="Path to model.")
parser.add_argument("target_pos", type=float, help="Target position.")
parser.add_argument(
    "-a", "--atol", default=0.007, type=float, help="Absolute error tolerance."
)

args = parser.parse_args()

model_path = args.model
target_pos = args.target_pos
atol = args.atol

model = PPO.load(model_path)

env = ch.Chameleon(target_pos=target_pos, atol=atol, train=False)

pos_hist = []
state = env.reset()
done = False
while not done:
    pos_hist.append(env.pos)
    action = model.predict(state, deterministic=True)[0]
    state, reward, done, _ = env.step(action)
    if done:
        try:
            pos_hist.append(env.winning_pos)
        except:
            print("didn't win :<()")

i = 1
for p in pos_hist:
    plt.scatter(p, 0 * p)
    plt.scatter(target_pos, 0)
    plt.scatter(0.1, 0)
    plt.title("Tip position %.3f" % p[-1])
    plt.savefig(f"frame{i}.png")
    plt.clf()
    i += 1


# history = []
# for i in range(len(pos_hist)):
#     x_p = pos_hist[i]
#     y = np.zeros(env.n_elems)
#     ar = np.vstack((x_p, y)).T
#     history.append(ar)

# with open("winning_history.txt", "wb") as fp:
#     pickle.dump(history, fp)
