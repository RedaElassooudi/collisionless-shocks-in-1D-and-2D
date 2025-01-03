import sys

sys.path.append("./")
sys.path.append("../")
import matplotlib.pyplot as plt

from results import Results
import visualizations as vis
import numpy as np

# 1734888965 -> n+1
# 1734890323 -> n (was testing this)
# 1734884902 -> wave in Ex
# 1734890999 -> update E twice, B not

run = "Results/2024-12-26T19h02m36s"
run = "Results/2024-12-27T14h34m25s"
x_e = Results.read(f"{run}/", "x_e")
v_e = Results.read(f"{run}/", "v_e")
x_max = 1
# vis.animate_phase_space(x_e, v_e, x_max, run.split("/")[-1])

# n+1 cell faces
# dataE = Results.read("Results/2024-12-25T03h18m09s/", "E")
# dataB = Results.read("Results/2024-12-25T03h18m09s/", "B")
# dataJ = Results.read("Results/2024-12-25T03h18m09s/", "J")
# t = Results.read("Results/2024-12-25T03h18m09s/", "t")
dataE = Results.read("Results/2024-12-26T17h02m19s/", "E")
dataB = Results.read("Results/2024-12-26T17h02m19s/", "B")
dataJ = Results.read("Results/2024-12-26T17h02m19s/", "J")
t = Results.read("Results/2024-12-26T17h02m19s/", "t")

# n cells
# data = Results.read("Results/1734862082/", "E")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
(line1,) = ax1.plot([], [], "b-o", label="Ex")
(line2,) = ax2.plot([], [], "r-o", label="By")
(line3,) = ax3.plot([], [], "b-o", label="Jx")

# Set axis limits (adjust based on your data range)
ax1.set_xlim(0, len(dataE[0]) - 1)
# ma = max([np.max(Ek[:, 0]) for Ek in dataE])
# mi = min([np.min(Ek[:, 0]) for Ek in dataE])
ax1.set_ylim(np.min(dataE[0]) - 1e-5, np.max(dataE[0]) + 1e-5)
ax1.legend()

ax2.set_xlim(0, len(dataB[0]) - 1)
# ma = max([np.max(Bk[:, 1]) for Bk in dataB])
# mi = min([np.min(Bk[:, 1]) for Bk in dataB])
ax2.set_ylim(np.min(dataB[0]) - 1e-5, np.max(dataB[0]) + 1e-5)
ax2.legend()

ax3.set_xlim(0, len(dataJ[0]) - 1)
# ma = max([np.max(Ek[:, 0]) for Ek in dataJ])
# mi = min([np.min(Ek[:, 0]) for Ek in dataJ])
ax3.set_ylim(np.min(dataJ[0]) - 1e-5, np.max(dataJ[0]) + 1e-5)
ax3.legend()

frame = 0
try:
    while True:
        # Update the first subplot
        x1 = range(len(dataE[frame][:, 0]))
        y1 = dataE[frame][:, 0]
        line1.set_data(x1, y1)
        ax1.set_title(f"Time t = {t[frame]:.8f}")
        m1 = np.max(np.abs(dataE[frame]))
        ax1.set_ylim(-1.05 * m1 - 1e-5, 1.05 * m1 + 1e-5)

        # Update the second subplot
        x2 = range(len(dataB[frame][:, 1]))
        y2 = dataB[frame][:, 1]
        line2.set_data(x2, y2)
        m2 = np.max(np.abs(dataB[frame]))
        ax2.set_ylim(-1.05 * m2 - 1e-5, 1.05 * m2 + 1e-5)

        # Update the second subplot
        x3 = range(len(dataJ[frame][:, 0]))
        y3 = dataJ[frame][:, 0]
        line3.set_data(x3, y3)
        m3 = np.max(np.abs(dataJ[frame]))
        ax3.set_ylim(-1.05 * m3 - 1e-5, 1.05 * m3 + 1e-5)

        # Redraw the plots
        plt.pause(0.5)  # Pause for 0.5 seconds
        fig.canvas.draw()

        # Update frame, cycling through the data
        frame = (frame + 1) % len(dataE)

        # press enter to advance to next frame
        input()
except KeyboardInterrupt:
    print("Animation stopped by user.")
