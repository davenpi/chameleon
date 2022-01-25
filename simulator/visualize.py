import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pickle
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("target_pos", type=float, help="Target position.")

args = parser.parse_args()

target = args.target_pos

with open("winning_history.txt", "rb") as fp:
    history = pickle.load(fp)

h = np.array(history)
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(xlim=(0, h.max() + 0.1), ylim=(-0.2, 0.2))
scatter = ax.scatter(history[0][:, 0], history[0][:, 1])


def update(frame_number):
    scatter.set_offsets(history[frame_number])
    history_tip = history[frame_number][-1][0]
    ax.scatter(0.1, 0, c="mediumseagreen")
    ax.scatter(target, 0, c="darkorange")
    ax.set_title("Tip position is %.3f" % history_tip)
    return (scatter,)


anim = FuncAnimation(fig, update, interval=30, frames=len(history))
writervideo = animation.FFMpegWriter(fps=3)
anim.save("reach_return_viz.mp4", writer=writervideo)
plt.close()
