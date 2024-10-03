import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Analytical solution function
def analytical_solution(x, t):
    temperature = 0
    for n in range(1, 1000, 2):
        temperature += (1/n)*np.sin((n*np.pi*x/100))*np.exp((-0.835*(n*np.pi/100)**2)*t)
    return (2000/np.pi)*temperature

Ta = []
for i in range(0, 601, 100):
    Ta.append(analytical_solution(20, i))

Ta2 = []
for i in range(0, 601, 50):
    Ta2.append(analytical_solution(20, i))


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

dx4 = 10
dt4 = 20
Nx4 = int(L / dx4) + 1
Nt4 = int(t_final / dt4) + 1

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

T4 = np.zeros((Nx4, Nt4))
T4[:, 0] = T_initial  # Set initial temperature
T4[0, :] = T_boundary  # Set boundary temperature at x = 0
T4[-1, :] = T_boundary  # Set boundary temperature at x = L


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

alpha4 = k * dt4 / (2 * dx4**2)
for n in range(0, Nt4 - 1):
    A4 = np.eye(Nx4 - 2) * (1 + 2 * alpha4) - np.eye(Nx4 - 2, k=1) * alpha4 - np.eye(Nx4 - 2, k=-1) * alpha4
    b4 = T4[1:-1, n] + alpha4 * (T4[:-2, n] - 2 * T4[1:-1, n] + T4[2:, n])
    T4[1:-1, n + 1] = np.linalg.solve(A4, b4)

print(T4)

e1 = [abs(Ta[i]-list(T[1])[i]) for i in range(0,len(Ta))]
e2 = [abs(Ta2[i]-list(T2[1])[i]) for i in range(0,len(Ta2))]
e3 = [abs(Ta[i]-list(T3[2])[i]) for i in range(0,len(Ta))]
# e4 = [abs(Ta[i]-list(T4[5])[i]) for i in range(0,len(Ta))]


# Temperature distribution after 10 minutes
# temperature_distribution = T[:, -1]
x = np.linspace(0, L, Nx)
x2 = np.linspace(0, L, Nx2)
x3 = np.linspace(0, L, Nx3)
x4 = np.linspace(0, L, Nx4)

y = np.linspace(0, t_final, Nt)
y2 = np.linspace(0, t_final, Nt2)
y3 = np.linspace(0, t_final, Nt3)
y4 = np.linspace(0, t_final, Nt4)




# print(T[1][-1])
# print(T2[1][-1])
# print(T3[2][-1])

# T = np.rot90(T,3)


# Generate heatmap using seaborn
# plt.figure()
# ax = sns.heatmap(T, cmap='hot', annot=True, fmt='.1f', xticklabels=x, yticklabels=np.arange(0, 601, 100), cbar_kws={'label': 'Temperature (°C)'})
# ax.set_xlabel('x (cm)')
# ax.set_ylabel('Time (s)')
# ax.set_title('Heatmap of temperature distribution along the rod')
# plt.show()

# plt.plot(y, Ta, marker = 'o', label="Analytical solution")
# plt.plot(y, T[1], marker = '+', label="dx = 20 cm, dt = 100 s")
# # plt.plot(np.arange(0, 601, 100), e1, marker = '+', label='first error')
# plt.plot(y2, T2[1], marker = '1', label="dx = 20 cm, dt = 50 s")
# # plt.plot(np.arange(0, 601, 50), e2, marker = '1', label='second error')
# plt.plot(y3, T3[2], marker = 'x', label="dx = 10 cm, dt = 100 s")
# # plt.plot(np.arange(0, 601, 100), e3, marker = 'x', label='third error')
# plt.plot(y4, T4[2], marker = 'o', label="dx = 10 cm, dt = 20 s")
# plt.xlabel("Time (s)")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.title("Evolution of temperature over time at x = 20 cm")
# plt.show()

# plt.plot(np.arange(0, 601, 100), e1, marker = '+', label='absolute error-dx = 20 cm, dt = 100 s')
# plt.plot(np.arange(0, 601, 50), e2, marker = '1', label='absolute error-dx = 20 cm, dt = 50 s')
# plt.plot(np.arange(0, 601, 100), e3, marker = 'x', label='absolute error-dx = 10 cm, dt = 100 s')
# plt.xlabel("Time (s)")
# plt.ylabel("Temperature (°C)")
# plt.title("Evolution of temperature over time at x = 20 cm")
# plt.legend()
# plt.show()


# # Plot the temperature distribution
# plt.figure()
# plt.plot(x, T[:, -7], marker = 'o', label='t = 0')
# plt.plot(x, T[:, -6], marker = '+', label='t = 100')
# plt.plot(x, T[:, -5], marker = '4', label='t = 200')
# plt.plot(x, T[:, -4], marker = 'x', label='t = 300')
# plt.plot(x, T[:, -3], marker = '1', label='t = 400')
# plt.plot(x, T[:, -2], marker = '2', label='t = 500')
# plt.plot(x, T[:, -1], marker = '3', label='t = 600')
# plt.xlabel("x (cm)")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.show()


# [  0.         0.           0.           0.           0.           0.           0.        ]
# [500.         412.93997414 353.46779462 310.49200877 277.62439361 251.16561877 228.95517637]
# [500.         491.77191606 471.25028472 444.92526608 416.40823175 387.67454904 359.77238346]
# [500.         491.77191606 471.25028472 444.92526608 416.40823175 387.67454904 359.77238346]
# [500.         412.93997414 353.46779462 310.49200877 277.62439361 251.16561877 228.95517637]
# [  0.         0.           0.           0.           0.           0.           0.        ]



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.linalg import solve_banded

# # Parameters and initial conditions
# L = 100.0
# kappa = 0.835
# t = 600
# T_initial = 500

# # Grid settings
# N = 6
# M = 6
# dx = L / (N-1)
# dt = t / (M-1)
# x = np.linspace(0, L, N)

# # Initialize temperature distribution
# u = np.zeros((M+1, N))
# u[:, 1:5] = T_initial
# # print(u)

# # Crank-Nicolson method
# r = kappa * dt / (2 * dx**2)

# # Set up the tridiagonal matrix
# A_upper = -r * np.ones(N - 1)
# A_middle = (1 + 2 * r) * np.ones(N)
# A_lower = -r * np.ones(N - 1)
# # print(A_upper)
# # print(A_middle)
# # print(A_lower)

# A = np.zeros((3, N))
# print(A)
# A[0, 1:] = A_upper
# print(A)
# A[1, :] = A_middle
# print(A)
# A[2, :-1] = A_lower
# print(A)

# # Time-stepping loop
# for j in range(0, M):
#     # Set up the right-hand side of the equation
#     b = np.zeros(N)
#     b[0] = r * 0
#     b[-1] = r * 0
#     b[1:-1] = u[j, 1:-1] + r * (u[j, :-2] - 2 * u[j, 1:-1] + u[j, 2:])

#     # Solve the linear system
#     u[j + 1, :] = solve_banded((1, 1), A, b)
# print(u)

# # for n in range(M):
# #     b = u[:, n].copy()
# #     b[1:-1] += r * (u[:-2, n] - 2 * u[1:-1, n] + u[2:, n])
# #     u[:, n + 1] = solve_banded((1, 1), A, b)
# # print(u)

# # # Plot the temperature distribution
# # plt.figure()
# # plt.plot(x, u[0, :], 'k--', label='t = 0')
# # plt.plot(x, u[1, :], 'r--', label='t = 100')
# # plt.plot(x, u[2, :], 'b--', label='t = 200')
# # plt.plot(x, u[3, :], 'g--', label='t = 300')
# # plt.plot(x, u[4, :], 'y--', label='t = 400')
# # plt.plot(x, u[5, :], 'c--', label='t = 500')
# # plt.plot(x, u[6, :], 'm--', label='t = 600')
# # plt.xlabel('x')
# # plt.ylabel('Temperature')
# # plt.legend()
# # plt.show()

