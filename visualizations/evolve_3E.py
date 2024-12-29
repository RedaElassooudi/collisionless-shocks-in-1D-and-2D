import sys

sys.path.append("./")
sys.path.append("../")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from results import Results
import numpy as np

run = "Results/2024-12-29T00h58m04s"

dataE = Results.read(f"{run}/", "E")
dataB = Results.read(f"{run}/", "B")
dataJ = Results.read(f"{run}/", "J")
t = Results.read(f"{run}/", "t")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
(line1,) = ax1.plot([], [], "b-o", label="Ex")
(line2,) = ax2.plot([], [], "r-o", label="By")
(line3,) = ax3.plot([], [], "b-o", label="Jx")

# Set up axis limits and labels
ax1.set_xlim(0, len(dataE[0]) - 1)
ax1.set_ylim(min(np.min(Ek[:, 0]) for Ek in dataE), max(np.max(Ek[:, 0]) for Ek in dataE))
ax1.legend()

ax2.set_xlim(0, len(dataB[0]) - 1)
ax2.set_ylim(min(np.min(Bk[:, 1]) for Bk in dataB), max(np.max(Bk[:, 1]) for Bk in dataB))
ax2.legend()

ax3.set_xlim(0, len(dataJ[0]) - 1)
ax3.set_ylim(min(np.min(Jk[:, 0]) for Jk in dataJ), max(np.max(Jk[:, 0]) for Jk in dataJ))
ax3.legend()


# Update function for FuncAnimation
def update(frame):
    # Update data for the first subplot
    x1 = range(len(dataE[frame][:, 0]))
    y1 = dataE[frame][:, 0]
    line1.set_data(x1, y1)
    ax1.set_title(f"Time t = {t[frame]:.4f}")

    # Update data for the second subplot
    x2 = range(len(dataB[frame][:, 1]))
    y2 = dataB[frame][:, 1]
    line2.set_data(x2, y2)

    # Update data for the third subplot
    x3 = range(len(dataJ[frame][:, 0]))
    y3 = dataJ[frame][:, 0]
    line3.set_data(x3, y3)

    return line1, line2, line3


# Create the animation
ani = FuncAnimation(fig, update, frames=len(dataE), interval=500, blit=True)

# Save the animation as a GIF
ani.save("dynamic_visualization.gif", writer="pillow", fps=2)

plt.show()
