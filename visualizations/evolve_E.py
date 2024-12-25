import matplotlib.pyplot as plt
import time

from results import Results
import visualizations as vis
import numpy as np

# Example data: a list of lists, where each inner list represents the data at a time step
# 1734888965 -> n+1
# 1734890323 -> n

x_e = Results.read("Results/1734890323/", "x_e")
v_e = Results.read("Results/1734890323/", "v_e")
x_max = 1
vis.animate_phase_space(x_e, v_e, x_max)

n_e = Results.read("Results/1734890323/", "n_e")
n_i = Results.read("Results/1734890323/", "n_i")
t = Results.read("Results/1734890323/", "t")
x = Results.read("Results/1734890323/", "x")
time_steps = [0, len(t) - 1]
vis.density_profiles_1D(time_steps, x, n_e, n_i, t)

input()

# n+1 cell faces
# data = Results.read("Results/1734889537/", "B")
data = Results.read("Results/1734890323/", "E")
# n cells
# data = Results.read("Results/1734862082/", "E")

# Initialize the figure and axis
fig, ax = plt.subplots()
(line,) = ax.plot([], [], "b-o")

ma = max([np.max(Ek) for Ek in data])
mi = min([np.min(Ek) for Ek in data])
print(ma)
print(mi)
# Set axis limits (adjust based on your data range)
ax.set_xlim(0, len(data[0]) - 1)  # x-axis range
ax.set_ylim(0.1, -0.1)  # y-axis range
# ax.set_ylim(ma, mi)  # y-axis range

# Infinite loop for plotting
frame = 0
try:
    while True:
        # Update the line data
        x = range(len(data[frame]))
        y = data[frame][:, 0]
        line.set_data(x, y)

        # Redraw the plot
        plt.pause(0.5)  # Pause for 0.5 seconds to simulate animation
        fig.canvas.draw()

        # Update frame, cycling through the data
        frame = (frame + 1) % len(data)
except KeyboardInterrupt:
    print("Animation stopped by user.")
