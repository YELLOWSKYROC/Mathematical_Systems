import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # Stability condition check
# if dt > (dx**2) / (2 * kappa):
#     raise ValueError("Stability condition not satisfied: reduce dt or increase dx")

# Analytical solution function
def analytical_solution(x, t):
    temperature = 0
    for n in range(1, 1000, 2):
        temperature += (1/n)*np.sin((n*np.pi*x/100))*np.exp((-0.835*(n*np.pi/100)**2)*t)
    return (2000/np.pi)*temperature

T = []
for i in range(0, 601, 100):
    T.append(analytical_solution(20, i))

# Parameters and Initial Conditions
L = 100.0
t = 600
kappa = 0.835
T_initial = 500

# grid settings
N1 = 6
M1 = 6
dx1 = L / (N1-1)
dt1 = t / (M1-1)
x1 = np.linspace(0, L, N1)

N2 = 6
M2 = 13
dx2 = L / (N2-1)
dt2 = t / (M2-1)
x2 = np.linspace(0, L, N2)

N3 = 11
M3 = 6
dx3 = L / (N3-1)
dt3 = t / (M3-1)
x3 = np.linspace(0, L, N3)

# Initialize the temperature profile
u1 = np.zeros((M1+1, N1))
u1[:, 1:5] = T_initial

u2 = np.zeros((M2+1, N2))
u2[:, 1:5] = T_initial

u3 = np.zeros((M3+1, N3))
u3[:, 1:10] = T_initial

# explicit finite difference method
for j in range(0, M1): # time step
    for i in range(1, N1-1): # space step
        u1[j + 1, i] = u1[j, i] + kappa * dt1 / dx1**2 * (u1[j, i - 1] - 2 * u1[j, i] + u1[j, i + 1])

for j in range(0, M2): # time step
    for i in range(1, N2-1): # space step
        u2[j + 1, i] = u2[j, i] + kappa * dt2 / dx2**2 * (u2[j, i - 1] - 2 * u2[j, i] + u2[j, i + 1])

for j in range(0, M3): # time step
    for i in range(1, N3-1): # space step
        u3[j + 1, i] = u3[j, i] + kappa * dt3 / dx3**2 * (u3[j, i - 1] - 2 * u3[j, i] + u3[j, i + 1])

print(u1)

# #  plot the temperature distribution along the bar at 100 s intervals
# plt.plot(x1, u1[0, :], label="T = 0 s")
# plt.plot(x1, u1[1, :], label="T = 100 s")
# plt.plot(x1, u1[2, :], label="T = 200 s")
# plt.plot(x1, u1[3, :], label="T = 300 s")
# plt.plot(x1, u1[4, :], label="T = 400 s")
# plt.plot(x1, u1[5, :], label="T = 500 s")
# plt.plot(x1, u1[6, :], label="T = 600 s")
# plt.xlabel("x (cm)")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.show()

# #  plot the temperature distribution along the bar at 100 s intervals
# plt.plot(x3, u3[0, :], label="T = 0 s")
# plt.plot(x3, u3[1, :], label="T = 100 s")
# plt.plot(x3, u3[2, :], label="T = 200 s")
# plt.plot(x3, u3[3, :], label="T = 300 s")
# plt.plot(x3, u3[4, :], label="T = 400 s")
# plt.plot(x3, u3[5, :], label="T = 500 s")
# plt.plot(x3, u3[6, :], label="T = 600 s")
# plt.xlabel("x (cm)")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.title("Temperature distribution along the rod at at 100 s intervals")
# plt.show()

# # plot the evolution of temperature over time at x = 20 cm
# x_axis1 = np.linspace(0, t, M1+1)
# x_axis2 = np.linspace(0, t, M2+1)
# x_axis3 = np.linspace(0, t, M3+1)
# plt.plot(x_axis1, T, label="Analytical solution")
# plt.plot(x_axis1, u1[:, 1], label="dx = 20 cm, dt = 100 s")
# plt.plot(x_axis2, u2[:, 1], label="dx = 20 cm, dt = 50 s")
# plt.plot(x_axis3, u3[:, 2], label="dx = 10 cm, dt = 100 s")
# plt.xlabel("Time (s)")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.title("Evolution of temperature over time at x = 20 cm")
# plt.show()

# # Generate heatmap using seaborn
# plt.figure()
# ax = sns.heatmap(u1, cmap='hot', annot=True, fmt='.1f', xticklabels=x1, yticklabels=np.arange(0, 601, 100), cbar_kws={'label': 'Temperature (°C)'})
# ax.set_xlabel('x (cm)')
# ax.set_ylabel('Time (s)')
# ax.set_title('Heatmap of temperature distribution along the rod')
# plt.show()