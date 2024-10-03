import numpy as np
import matplotlib.pyplot as plt

# Analytical solution function
def analytical_solution(x, t):
    temperature = 0
    for n in range(1, 1000, 2):
        temperature += (1/n)*np.sin((n*np.pi*x/100))*np.exp((-0.835*(n*np.pi/100)**2)*t)
    return (2000/np.pi)*temperature

Ta = []
for i in range(0, 601, 100):
    Ta.append(analytical_solution(20, i))


# Constants
L = 100  # Length of the rod in cm
k = 0.835  # Thermal conductivity
T_initial = 500  # Initial temperature in Celsius
T_boundary = 0  # Boundary temperature in Celsius
t_final = 600  # Time in seconds

# Discretization
dx = 20  # Spatial step in cm
dt = 100  # Time step in seconds
Nx = int(L / dx) + 1  # Number of spatial points
Nt = int(t_final / dt) + 1  # Number of time points


dx2 = 20  # Spatial step in cm
dt2 = 50  # Time step in seconds
Nx2 = int(L / dx2) + 1  # Number of spatial points
Nt2 = int(t_final / dt2) + 1  # Number of time points

dx3 = 10  # Spatial step in cm
dt3 = 100  # Time step in seconds
Nx3 = int(L / dx3) + 1  # Number of spatial points
Nt3 = int(t_final / dt3) + 1  # Number of time points

# Initialize temperature array
T = np.zeros((Nx, Nt))
T[:, 0] = T_initial  # Set initial temperature
T[0, :] = T_boundary  # Set boundary temperature at x = 0
T[-1, :] = T_boundary  # Set boundary temperature at x = L

T2 = np.zeros((Nx2, Nt2))
T2[:, 0] = T_initial  # Set initial temperature
T2[0, :] = T_boundary  # Set boundary temperature at x = 0
T2[-1, :] = T_boundary  # Set boundary temperature at x = L

T3 = np.zeros((Nx3, Nt3))
T3[:, 0] = T_initial  # Set initial temperature
T3[0, :] = T_boundary  # Set boundary temperature at x = 0
T3[-1, :] = T_boundary  # Set boundary temperature at x = L



# Crank-Nicolson method
alpha = k * dt / (2 * dx**2)
for n in range(0, Nt - 1):
    A = (np.eye(Nx - 2) * (1 + 2 * alpha)) - (np.eye(Nx - 2, k=1) * alpha) - (np.eye(Nx - 2, k=-1) * alpha)
    b = T[1:-1, n] + alpha * (T[:-2, n] - 2 * T[1:-1, n] + T[2:, n])
    T[1:-1, n + 1] = np.linalg.solve(A, b)

alpha2 = k * dt2 / (2 * dx2**2)
for n in range(0, Nt2 - 1):
    A2 = np.eye(Nx2 - 2) * (1 + 2 * alpha2) - np.eye(Nx2 - 2, k=1) * alpha2 - np.eye(Nx2 - 2, k=-1) * alpha2
    b2 = T2[1:-1, n] + alpha2 * (T2[:-2, n] - 2 * T2[1:-1, n] + T2[2:, n])
    T2[1:-1, n + 1] = np.linalg.solve(A2, b2)

alpha3 = k * dt3 / (2 * dx3**2)
for n in range(0, Nt3 - 1):
    A3 = np.eye(Nx3 - 2) * (1 + 2 * alpha3) - np.eye(Nx3 - 2, k=1) * alpha3 - np.eye(Nx3 - 2, k=-1) * alpha3
    b3 = T3[1:-1, n] + alpha3 * (T3[:-2, n] - 2 * T3[1:-1, n] + T3[2:, n])
    T3[1:-1, n + 1] = np.linalg.solve(A3, b3)


# Temperature distribution after 10 minutes
# temperature_distribution = T[:, -1]
x = np.linspace(0, L, Nx)
x2 = np.linspace(0, L, Nx2)
x3 = np.linspace(0, L, Nx3)

y = np.linspace(0, t_final, Nt)
y2 = np.linspace(0, t_final, Nt2)
y3 = np.linspace(0, t_final, Nt3)

plt.plot(y, Ta, label="Analytical solution")
plt.plot(y, T[1], label="dx = 20 cm, dt = 100 s")
plt.plot(y2, T2[1], label="dx = 20 cm, dt = 50 s")
plt.plot(y3, T3[2], label="dx = 10 cm, dt = 100 s")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("Evolution of temperature over time at x = 20 cm")
plt.show()

# # Plot the temperature distribution
# plt.figure()
# plt.plot(x, T[:, -7], 'k--', label='t = 0')
# plt.plot(x, T[:, -6], 'r--', label='t = 100')
# plt.plot(x, T[:, -5], 'b--', label='t = 200')
# plt.plot(x, T[:, -4], 'g--', label='t = 300')
# plt.plot(x, T[:, -3], 'y--', label='t = 400')
# plt.plot(x, T[:, -2], 'c--', label='t = 500')
# plt.plot(x, T[:, -1], 'm--', label='t = 600')
# plt.xlabel('x')
# plt.ylabel('Temperature')
# plt.legend()
# plt.show()

# [  0.         0.           0.           0.           0.           0.           0.        ]
# [500.         412.93997414 353.46779462 310.49200877 277.62439361 251.16561877 228.95517637]
# [500.         491.77191606 471.25028472 444.92526608 416.40823175 387.67454904 359.77238346]
# [500.         491.77191606 471.25028472 444.92526608 416.40823175 387.67454904 359.77238346]
# [500.         412.93997414 353.46779462 310.49200877 277.62439361 251.16561877 228.95517637]
# [  0.         0.           0.           0.           0.           0.           0.        ]

