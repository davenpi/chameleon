import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pickle
import numpy as np

with open("simulator/winning_history.txt", "rb") as fp:
    history = pickle.load(fp)

h = np.array(history)
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(xlim=(0, h.max() + 0.1), ylim=(-0.2, 0.2))
ax.scatter(0.18, 0)  # this is really dumb. need to automate this
ax.scatter(0.1, 0)
scatter = ax.scatter(history[0][:, 0], history[0][:, 1])


def update(frame_number):
    scatter.set_offsets(history[frame_number])
    history_tip = history[frame_number][-1][0]
    ax.set_title(f"Tip position is {history_tip}")
    return (scatter,)


anim = FuncAnimation(fig, update, interval=30, frames=len(history))
writervideo = animation.FFMpegWriter(fps=3)
anim.save("reach_return_viz.mp4", writer=writervideo)
plt.close()
