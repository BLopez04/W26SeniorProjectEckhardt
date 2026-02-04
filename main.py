import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

def equation(t):
    return 0.1 * t * (10 - t)


time_values = np.linspace(0, 10, 400)
x_values = equation(time_values)

c_ref_t = np.linspace(0, 10, 400)
c_ref_x = c_ref_t

plt.figure(figsize=(8, 6))
plt.plot(x_values, time_values, label=r"x(t) = -t^2", color="blue", linewidth=2)

plt.plot(c_ref_x, c_ref_t, label=r"x(t) = t", color="red", linewidth=2)

plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("x: distance from origin")
plt.ylabel("t: time")

plt.title("Comparison: Linear vs Parabolic Worldlines")

plt.legend()

plt.show()
