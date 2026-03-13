import numpy as np
import matplotlib
from scipy.optimize import fsolve
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



# Calculate time difference from start to end
# Mark ticks that happen every dt on the y axis stationary.
# Count ticks on blue axis and y axis.

# Get rid of that weird artifact at the last second of the sinusoidal line (where it moves up)


# Add a sinusoidal curve that has a greater "hump"


plt.rc('font', size=15)

# Parameters
n = 5
TMax = 100  # max time
A = 0.6     # fraction of the speed of light
dt_emit = 1.0

# Ship Worldline 1 (sinusoidal)
def equation(t):
    return (A * TMax / np.pi) * np.sin(np.pi * t / TMax)
def velocity(t):
    return A * np.cos(np.pi * t / TMax)

def gamma(t):
    v = velocity(t)
    return 1.0 / np.sqrt(1 - v**2)

# Ship Worldline 2 (Sharp angle turn)
def equation_turn(t):
    return np.where(t < TMax/2, A * t, A * (TMax - t))

def velocity_turn(t):
    return np.where(t < TMax/2, A, -A)

def gamma_turn(t):
    v = velocity_turn(t)
    return 1.0 / np.sqrt(1 - v**2)

# Light pulse arrival time
def arrival_time(t_s, x_s):
    return t_s + x_s

# Time grid

t_grid = np.linspace(0, TMax, 400)
dt = t_grid[1] - t_grid[0] # difference in emission times
#proper_time = np.cumsum(dt / gamma(t_grid))

# ----------------------------------------
# Wordline 1
# ----------------------------------------

# Emit one pulse every Δt in coordinate time

t_emit = np.arange(0, TMax, dt_emit)
t_emit_turn = t_emit.copy()

# positions at emission times
x_emit = equation(t_emit)
arrival_times = t_emit + x_emit
arrival_spacing = np.diff(arrival_times)

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
plt.plot(t_grid, t_grid, label=r"Speed of Light worldline", color="red", linewidth=2)


plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("x: distance from origin")
plt.ylabel("t: time")

plt.title("Minkowski Diagram - Sinusoidal Worldline")
plt.legend()

# Figure 1B - Arrival-time spacing (Doppler effect)

plt.figure(figsize=(8, 8))

doppler_factor = arrival_spacing / dt_emit

# arrival spacing vs emission time
plt.plot(arrival_times[:-1], doppler_factor,
         color="black", linewidth=2, label="Doppler factor k(t)")

# speed of light reference with constant spacing
plt.axhline(1.0, color="red", linestyle="--", linewidth=2,
            label="k = 1 (stationary source)")

plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlabel("Arrival time")
plt.ylabel("Doppler factor k")
plt.title("Relativistic Doppler Factor — Sinusoidal Worldline")
plt.legend()

# ----------------------------------------
# Wordline 2
# ----------------------------------------
# Values for linear line

# positions at emission times
x_emit_turn = equation_turn(t_emit_turn)
arrival_times_turn = t_emit_turn + x_emit_turn
arrival_spacing_turn = np.diff(arrival_times_turn)

#proper_time_turn = np.cumsum(dt / gamma_turn(t_grid))
## Figure 2A- Minkowski diagram for sharp turn line

plt.figure(figsize=(8, 8))
for i in range(0, len(t_emit_turn), n):
    t_s = t_emit_turn[i]
    x_s = x_emit_turn[i]
    t_arr = arrival_time(t_s, x_s)
    plt.plot([x_s, 0], [t_s, t_arr], color="green", alpha=0.6)

# Ship Worldline 2
plt.plot([equation_turn(t) for t in t_grid], t_grid, label=r"Ship Worldline 2 (Linear with turn)", color="blue", linewidth=2)
# Speed of light worldline
plt.plot(t_grid, t_grid, label=r"Speed of Light worldline", color="red", linewidth=2)

plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("x: distance from origin")
plt.ylabel("t: time")

plt.title("Minkowski Diagram - Sharp turn Worldline")
plt.legend()

# Figure 1B - Arrival-time spacing (Doppler effect)

plt.figure(figsize=(8, 8))

doppler_factor_turn = arrival_spacing_turn / dt_emit

# arrival spacing vs emission time
plt.plot(arrival_times_turn[:-1], doppler_factor_turn,
         color="black", linewidth=2,
         label="Doppler factor k(t)")


plt.axhline(1.0, color="red", linestyle="--", linewidth=2,
            label="k = 1 (stationary source)")

plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlabel("Arrival time")
plt.ylabel("Doppler factor k")
plt.title("Relativistic Doppler Factor — 90° Turn Worldline")
plt.legend()

plt.show()