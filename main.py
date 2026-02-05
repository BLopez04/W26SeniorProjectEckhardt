import numpy as np
import matplotlib
from scipy.optimize import fsolve
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

ax_factor = 0
init_guess = [1,1]

def equation(t):
    return 0.1 * t * (10 - t)

def line_eq(vars):
    x, y = vars
    spd_light = y-x - ax_factor # speed of light line traversing up the t axis
    given_eq = y-x-0.1*y**2 # given equation
    return [spd_light, given_eq]

time_values = np.linspace(0, 10, 400)
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
dist_lst = np.zeros(len(line_t))

indx =0
for i in range(0, len(pts_x), 2):
    a,b = pts_lst[i], pts_lst[i+1]
    # plt.plot(pts_x[i:i+2], pts_t[i:i+2], color='green', label = "inner area defined by 45 degree lines")
    dist = np.linalg.norm(a-b)
    dist_lst[indx] = dist
    indx += 1

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

plt.figure(figsize=(8, 6))
plt.plot(time_values, dist_lst, label="Distance from observer", color="black", linewidth=2)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("t: time")
plt.ylabel("d: relative distance from observer to equation")

plt.title("Comparison: Linear vs Parabolic Worldlines")
plt.legend()
plt.show()
