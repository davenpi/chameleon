import os
import pickle
import argparse
import numpy as np
import chameleon as ch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO, DDPG

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

model = DDPG.load(model_path)

env = ch.Chameleon(target_pos=target_pos, atol=atol, train=False, E=50)

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

# i = 1
# os.mkdir("framebyframe")
# for p in pos_hist:
#     plt.scatter(p, 0 * p)
#     plt.scatter(target_pos, 0)
#     plt.scatter(1, 0)
#     plt.title("Tip position %.3f" % p[-1])
#     plt.savefig(f"framebyframe/frame{i}.png")
#     plt.clf()
#     i += 1


history = []
for i in range(len(pos_hist)):
    x_p = pos_hist[i]
    y = np.zeros(env.n_elems)
    ar = np.vstack((x_p, y)).T
    history.append(ar)

h = np.array(history)
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(xlim=(0, h.max() + 0.1), ylim=(-0.2, 0.2))
scatter = ax.scatter(history[0][:, 0], history[0][:, 1])


def update(frame_number):
    scatter.set_offsets(history[frame_number])
    history_tip = history[frame_number][-1][0]
    ax.scatter(env.pos_init[-1], 0, c="mediumseagreen")
    ax.scatter(target_pos, 0, c="darkorange")
    ax.set_title(
        f"Tip position is %.3f. initial target is {env.target_pos}" % history_tip
    )
    return (scatter,)


anim = FuncAnimation(fig, update, interval=30, frames=len(history))
writervideo = animation.FFMpegWriter(fps=3)
anim.save("reach_return_viz.mp4", writer=writervideo)
plt.close()
