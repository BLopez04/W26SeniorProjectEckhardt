import numpy as np
import matplotlib
from scipy.optimize import fsolve
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
stuff = 0
init_guess = [1,1]

def equation(t):
    return 0.1 * t * (10 - t)

def line_eq(vars):
    x, y = vars
    eq1 = y-x - stuff # we ball with scoping around here
    eq2 = y-x-0.1*y**2
    return [eq1, eq2]

time_values = np.linspace(0, 10, 400)
x_values = equation(time_values)

# list of intersections between equation and spd of light line
inter_lst = np.zeros((len(time_values),2))

line_t = time_values.copy() # timestep values for t axis

print("intersection stuff: ")
indx = 0
for item in line_t:
    stuff = item
    intersection = fsolve(line_eq, init_guess)
    inter_lst[indx] = intersection
    indx += 1

cpy_inter = inter_lst.copy() # don't mind this, testing different methods here

indx = 0
for i in line_t:
    stuff = i
    intersection = fsolve(line_eq, [0,0])
    inter_lst[indx] = intersection
    indx += 1

pts_x = np.array([])
pts_t = np.array([])
for i in range(len(line_t)):
    x1, t1 = 0, line_t[i]
    x2, t2 = cpy_inter[i]
    pts_x = np.append(pts_x, x1)
    pts_x = np.append(pts_x, x2)
    pts_t = np.append(pts_t, t1)
    pts_t = np.append(pts_t, t2)

for i in range(0, len(pts_x), 2):
    plt.plot(pts_x[i:i+2], pts_t[i:i+2])

# speed of light line
c_ref_t = time_values.copy()
c_ref_x = c_ref_t


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
