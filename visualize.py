import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pickle
import numpy as np

with open("simulator/history.txt", "rb") as fp:
    history = pickle.load(fp)

h = np.array(history)
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(xlim=(0, h.max() + 1), ylim=(-0.2, 0.2))
scatter = ax.scatter(history[0][:, 0], history[0][:, 1])


def update(frame_number):
    scatter.set_offsets(history[frame_number])
    return (scatter,)


anim = FuncAnimation(fig, update, interval=10, frames=len(history))
writervideo = animation.FFMpegWriter(fps=120)
anim.save("system_viz.mp4", writer=writervideo)
plt.close()