import numpy as np
import matplotlib
from scipy.optimize import fsolve
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

n = 50
ax_factor = 0
init_guess = [1,1]

def equation(t):
    return 0.1 * t * (10 - t)

def line_eq(vars):
    x, y = vars
    spd_light = y + x - ax_factor # speed of light line traversing up the t axis
    given_eq = y - x - 0.1 * y ** 2 # given equation
    return [spd_light, given_eq]

time_values = np.linspace(0, 10, 400)
# difference in emission times
dt_emit = time_values[1] - time_values[0]
x_values = equation(time_values)
line_t = time_values.copy() # timestep values for t axis

# list of intersections between equation and spd of light line
inter_lst = np.zeros((len(time_values),2))

indx = 0
for i in line_t:
    ax_factor = i #increment time displacement
    intersection = fsolve(line_eq, [0,0])
    inter_lst[indx] = intersection
    indx += 1

pts_x = np.array([])
pts_t = np.array([])
for i in range(len(line_t)):
    x1, t1 = 0, line_t[i]
    x2, t2 = inter_lst[i]
    pts_x = np.append(pts_x, x1)
    pts_x = np.append(pts_x, x2)
    pts_t = np.append(pts_t, t1)
    pts_t = np.append(pts_t, t2)


pts_x_col, pts_t_col = pts_x.reshape(-1,1), pts_t.reshape(-1,1)
pts_lst = np.concatenate((pts_x_col, pts_t_col), axis=1)

## Figure 1 - Minkowski diagram

indx = 0
for i in range(0, len(pts_x), 2 * n):
    plt.plot(pts_x[i:i+2], pts_t[i:i+2],
             color='green', alpha=0.6)


# Ship worldline
plt.plot(x_values, time_values, label=r"Ship worldline", color="blue", linewidth=2)
# Speed of light worldline
plt.plot(time_values, time_values, label=r"Speed of Light worldline: x(t) = t", color="red", linewidth=2)

arrival_times = time_values + x_values

arrival_spacing = np.diff(arrival_times)

plt.scatter(
    np.zeros_like(arrival_times[::n]),
    arrival_times[::n],
    color="black",
    s=15,
    zorder=5,
    label="Pulse arrivals at origin"
)


plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("x: distance from origin")
plt.ylabel("t: time")

plt.title("Comparison: Linear vs Parabolic Worldlines")

plt.legend()

# Figure 2 - Arrival-time spacing (Doppler effect)

plt.figure(figsize=(8, 8))

arrival_times = time_values + x_values

arrival_spacing = np.diff(arrival_times)

# arrival spacing vs emission time
plt.plot(time_values[1:], arrival_spacing,
         color="black", linewidth=2, label="Arrival-time spacing (Doppler effect)")

# speed of light reference with constant spacing
plt.axhline(dt_emit,
            color="red", linestyle="--", linewidth=2,
            label="Constant spacing (stationary source)")

plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

plt.xlabel("s: emission time")
plt.ylabel("t': arrival spacing")

plt.title("Pulse Arrival-Time Spacing (Doppler Effect)")
plt.legend()
plt.show()