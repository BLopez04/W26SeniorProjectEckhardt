import numpy as np
import matplotlib
from scipy.optimize import fsolve
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Add Lorentz Factor to space out the green lines based on velocity of the ship
# Lorentz factor based off of the velocity of the ship (the position derivative)
# Add another equation of a ship that has a 90 degree right angle turn back to origin

# Parameters
n = 25
TMax = 100 # max time
A = 22 # amplitude

# Ship Worldline 1 (sinusoidal)
def equation(t):
    return A * np.sin(np.pi * t / TMax)

def velocity(t):
    return A * (np.pi / TMax) * np.cos(np.pi * t / TMax)

def gamma(t):
    v = velocity(t)
    return 1.0 / np.sqrt(1 - v**2)

# Ship Worldline 2 (Sharp angle turn)
def equation_turn(t):
    if t < TMax/2:
        return A * (t / (TMax/2)) # move away
    else:
        return A * (1 - (t - TMax/2) / (TMax/2)) # return back

# Light pulse arrival time
def arrival_time(t_s, x_s):
    return t_s + x_s

# Time grid

t_grid = np.linspace(0, TMax, 400)
dt = t_grid[1] - t_grid[0] # difference in emission times

gamma_vals = gamma(t_grid)
proper_time = np.cumsum(dt / gamma_vals)

# Emission events at equal intervals
tau_emit = np.linspace(0, proper_time[-1], 400)
t_emit = np.interp(tau_emit, proper_time, t_grid)

# ----------------------------------------
# Wordline 1
# ----------------------------------------

# positions at emission times
x_emit = equation(t_emit)
arrival_times = t_emit + x_emit

## Figure 1A - Minkowski diagram for sinusoidal line

plt.figure(figsize=(8, 8))
for i in range(0, len(t_emit), n):
    t_s = t_emit[i]
    x_s = x_emit[i]
    t_arr = arrival_time(t_s, x_s)
    plt.plot([x_s, 0], [t_s, t_arr], color="green", alpha=0.6)

# Ship Worldline 1
plt.plot(equation(t_grid), t_grid, label=r"Ship Worldline 1 (Sinusoidal)", color="blue", linewidth=2)
# Speed of light worldline
plt.plot(t_grid, t_grid, label=r"Speed of Light worldline: x(t) = t", color="red", linewidth=2)

# Pulse arrivals (only a subsample)

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

plt.title("Minkowski Diagram with Lorentz Factor and Sinusoidal turn")
plt.legend()

# Figure 1B - Arrival-time spacing (Doppler effect)

plt.figure(figsize=(8, 8))

arrival_spacing = np.diff(arrival_times)
# Stationary reference line
dt_proper = tau_emit[1] - tau_emit[0]
# arrival spacing vs emission time
plt.plot(t_emit[1:], arrival_spacing,
         color="black", linewidth=2, label="Arrival-time spacing (Doppler effect)")

# speed of light reference with constant spacing
plt.axhline(dt_proper,
            color="red", linestyle="--", linewidth=2,
            label="Constant spacing (stationary source)")

plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

plt.xlabel("s: emission time")
plt.ylabel("t': arrival spacing")

plt.title("Pulse Arrival-Time Spacing (Doppler Effect) on sinusoidal turn")
plt.legend()

# ----------------------------------------
# Wordline 2
# ----------------------------------------

# positions at emission times
x_emit_turn = np.array([equation_turn(t) for t in t_emit])
arrival_times_turn = t_emit + x_emit_turn

## Figure 2A- Minkowski diagram for sharp turn line

plt.figure(figsize=(8, 8))
for i in range(0, len(t_emit), n):
    t_s = t_emit[i]
    x_s = x_emit_turn[i]
    t_arr = arrival_time(t_s, x_s)
    plt.plot([x_s, 0], [t_s, t_arr], color="green", alpha=0.6)

# Ship Worldline 2
plt.plot([equation_turn(t) for t in t_grid], t_grid, label=r"Ship Worldline 1 (Sinusoidal)", color="blue", linewidth=2)
# Speed of light worldline
plt.plot(t_grid, t_grid, label=r"Speed of Light worldline: x(t) = t", color="red", linewidth=2)

# Pulse arrivals (only a subsample)

plt.scatter(
    np.zeros_like(arrival_times_turn[::n]),
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

plt.title("Minkowski Diagram with Lorentz Factor and 90° Turn")
plt.legend()

# Figure 1B - Arrival-time spacing (Doppler effect)

plt.figure(figsize=(8, 8))

arrival_spacing_turn = np.diff(arrival_times_turn)

# arrival spacing vs emission time
plt.plot(t_emit[1:], arrival_spacing_turn,
         color="black", linewidth=2, label="Arrival-time spacing (Doppler effect)")

# speed of light reference with constant spacing
plt.axhline(dt_proper,
            color="red", linestyle="--", linewidth=2,
            label="Constant spacing (stationary source)")

plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

plt.xlabel("s: emission time")
plt.ylabel("t': arrival spacing")

plt.title("Pulse Arrival-Time Spacing (Doppler Effect) on sharp turn")
plt.legend()

plt.show()