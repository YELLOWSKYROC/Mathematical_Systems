import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# g = 0.1
# c = 400
L = 100

T = []
for x in range(0, 101, 5):
    T.append(-x*0.1*(x-L)+400)

# print(T)


# arr = list(range(101))
# print(T[0])
# print(T[10])
# print(T[20])
# print(T[30])
# print(T[40])
# print(T[50])
# print(T[60])
# print(T[70])
# print(T[80])
# print(T[90])
# print(T[100])

# # # Plot temperature distribution
# # plt.plot(arr, T)
# # plt.xlabel('Position (cm)')
# # plt.ylabel('Temperature (°C)')
# # plt.title('Temperature Distribution after 10 Minutes')
# # plt.show()

# Constants
L = 100  # Length of the rod in cm
k = 0.835  # Thermal conductivity
T_initial = 500  # Initial temperature in Celsius
T_boundary = 0  # Boundary temperature in Celsius
t_final = 60  # Time in seconds
t_final2 = 600

dx23 = 20  # Spatial step in cm
dt23 = 10  # Time step in seconds
Nx23 = int(L / dx23) + 1  # Number of spatial points
Nt23 = int(t_final2 / dt23) + 1  # Number of time points

T23 = np.zeros((Nx23, Nt23))
T23[:, 0] = T_initial  # Set initial temperature
T23[0, :] = T_boundary  # Set boundary temperature at x = 0
T23[-1, :] = T_boundary  # Set boundary temperature at x = L


# # Discretization
# dx = 20  # Spatial step in cm
# dt = 20  # Time step in seconds
# Nx = int(L / dx) + 1  # Number of spatial points
# Nt = int(t_final / dt) + 1  # Number of time points

# dx2 = 10  # Spatial step in cm
# dt2 = 20  # Time step in seconds
# Nx2 = int(L / dx2) + 1  # Number of spatial points
# Nt2 = int(t_final / dt2) + 1  # Number of time points

dx3 = 20  # Spatial step in cm
dt3 = 10  # Time step in seconds
Nx3 = int(L / dx3) + 1  # Number of spatial points
Nt3 = int(t_final / dt3) + 1  # Number of time points

# dx4 = 10  # Spatial step in cm
# dt4 = 10  # Time step in seconds
# Nx4 = int(L / dx4) + 1  # Number of spatial points
# Nt4 = int(t_final / dt4) + 1  # Number of time points

# dx5 = 5  # Spatial step in cm
# dt5 = 10  # Time step in seconds
# Nx5 = int(L / dx5) + 1  # Number of spatial points
# Nt5 = int(t_final / dt5) + 1  # Number of time points

# dx6 = 10  # Spatial step in cm
# dt6 = 5  # Time step in seconds
# Nx6 = int(L / dx6) + 1  # Number of spatial points
# Nt6 = int(t_final / dt6) + 1  # Number of time points

# dx7 = 5  # Spatial step in cm
# dt7 = 5  # Time step in seconds
# Nx7 = int(L / dx7) + 1  # Number of spatial points
# Nt7 = int(t_final / dt7) + 1  # Number of time points


# # Initialize temperature array
# T1 = np.zeros((Nx, Nt))
# T1[1,0] = 560
# T1[2,0] = 640
# T1[3,0] = 640
# T1[4,0] = 560

# T2 = np.zeros((Nx2, Nt2))
# T2[1,0] = 490
# T2[2,0] = 560
# T2[3,0] = 610
# T2[4,0] = 640
# T2[5,0] = 650
# T2[6,0] = 640
# T2[7,0] = 610
# T2[8,0] = 560
# T2[9,0] = 490


T3 = np.zeros((Nx3, Nt3))
T3[1,0] = 560
T3[2,0] = 640
T3[3,0] = 640
T3[4,0] = 560

# T4 = np.zeros((Nx4, Nt4))
# T4[1,0] = 490
# T4[2,0] = 560
# T4[3,0] = 610
# T4[4,0] = 640
# T4[5,0] = 650
# T4[6,0] = 640
# T4[7,0] = 610
# T4[8,0] = 560
# T4[9,0] = 490

# T5 = np.zeros((Nx5, Nt5))
# for i in range(21):
#     T5[i,0] = T[i]

# T6 = np.zeros((Nx6, Nt6))
# T6[1,0] = 490
# T6[2,0] = 560
# T6[3,0] = 610
# T6[4,0] = 640
# T6[5,0] = 650
# T6[6,0] = 640
# T6[7,0] = 610
# T6[8,0] = 560
# T6[9,0] = 490

# T7 = np.zeros((Nx7, Nt7))
# for i in range(21):
#     T7[i,0] = T[i]






# # # Crank-Nicolson method
# alpha = k * dt / (2 * dx**2)
# for n in range(0, Nt - 1):
#     A = (np.eye(Nx - 2) * (1 + 2 * alpha)) - (np.eye(Nx - 2, k=1) * alpha) - (np.eye(Nx - 2, k=-1) * alpha)
#     b = T1[1:-1, n] + alpha * (T1[:-2, n] - 2 * T1[1:-1, n] + T1[2:, n])
#     T1[1:-1, n + 1] = np.linalg.solve(A, b)

# alpha2 = k * dt2 / (2 * dx2**2)
# for n in range(0, Nt2 - 1):
#     A2 = (np.eye(Nx2 - 2) * (1 + 2 * alpha2)) - (np.eye(Nx2 - 2, k=1) * alpha2) - (np.eye(Nx2 - 2, k=-1) * alpha2)
#     b2 = T2[1:-1, n] + alpha2 * (T2[:-2, n] - 2 * T2[1:-1, n] + T2[2:, n])
#     T2[1:-1, n + 1] = np.linalg.solve(A2, b2)

alpha3 = k * dt3 / (2 * dx3**2)
for n in range(0, Nt3 - 1):
    A3 = (np.eye(Nx3 - 2) * (1 + 2 * alpha3)) - (np.eye(Nx3 - 2, k=1) * alpha3) - (np.eye(Nx3 - 2, k=-1) * alpha3)
    b3 = T3[1:-1, n] + alpha3 * (T3[:-2, n] - 2 * T3[1:-1, n] + T3[2:, n])
    T3[1:-1, n + 1] = np.linalg.solve(A3, b3)

# print(T3)

# alpha4 = k * dt4 / (2 * dx4**2)
# for n in range(0, Nt4 - 1):
#     A4 = (np.eye(Nx4 - 2) * (1 + 2 * alpha4)) - (np.eye(Nx4 - 2, k=1) * alpha4) - (np.eye(Nx4 - 2, k=-1) * alpha4)
#     b4 = T4[1:-1, n] + alpha4 * (T4[:-2, n] - 2 * T4[1:-1, n] + T4[2:, n])
#     T4[1:-1, n + 1] = np.linalg.solve(A4, b4)

# alpha5 = k * dt5 / (2 * dx5**2)
# for n in range(0, Nt5 - 1):
#     A5 = (np.eye(Nx5 - 2) * (1 + 2 * alpha5)) - (np.eye(Nx5 - 2, k=1) * alpha5) - (np.eye(Nx5 - 2, k=-1) * alpha5)
#     b5 = T5[1:-1, n] + alpha5 * (T5[:-2, n] - 2 * T5[1:-1, n] + T5[2:, n])
#     T5[1:-1, n + 1] = np.linalg.solve(A5, b5)

# alpha6 = k * dt6 / (2 * dx6**2)
# for n in range(0, Nt6 - 1):
#     A6 = (np.eye(Nx6 - 2) * (1 + 2 * alpha6)) - (np.eye(Nx6 - 2, k=1) * alpha6) - (np.eye(Nx6 - 2, k=-1) * alpha6)
#     b6 = T6[1:-1, n] + alpha6 * (T6[:-2, n] - 2 * T6[1:-1, n] + T6[2:, n])
#     T6[1:-1, n + 1] = np.linalg.solve(A6, b6)

# alpha7 = k * dt7 / (2 * dx7**2)
# for n in range(0, Nt7 - 1):
#     A7 = (np.eye(Nx7 - 2) * (1 + 2 * alpha7)) - (np.eye(Nx7 - 2, k=1) * alpha7) - (np.eye(Nx7 - 2, k=-1) * alpha7)
#     b7 = T7[1:-1, n] + alpha7 * (T7[:-2, n] - 2 * T7[1:-1, n] + T7[2:, n])
#     T7[1:-1, n + 1] = np.linalg.solve(A7, b7)

alpha23 = k * dt23 / (2 * dx23**2)
for n in range(0, Nt23 - 1):
    A23 = (np.eye(Nx23 - 2) * (1 + 2 * alpha23)) - (np.eye(Nx23 - 2, k=1) * alpha23) - (np.eye(Nx23 - 2, k=-1) * alpha23)
    b23 = T23[1:-1, n] + alpha23 * (T23[:-2, n] - 2 * T23[1:-1, n] + T23[2:, n])
    T23[1:-1, n + 1] = np.linalg.solve(A23, b23)

# print(T23)

# Temperature distribution after 10 minutes
# temperature_distribution = T[:, -1]
# x = np.linspace(0, L, Nx)
# y = np.linspace(0, t_final, Nt)

# x2 = np.linspace(0, L, Nx2)
# y2 = np.linspace(0, t_final, Nt2)

x23 = np.linspace(0, L, Nx23)
y23 = np.linspace(0, t_final2, Nt23)

x3 = np.linspace(0, L, Nx3)
y3 = np.linspace(0, t_final, Nt3)

# x4 = np.linspace(0, L, Nx4)
# y4 = np.linspace(0, t_final, Nt4)

# x5 = np.linspace(0, L, Nx5)
# y5 = np.linspace(0, t_final, Nt5)

# x6 = np.linspace(0, L, Nx6)
# y6 = np.linspace(0, t_final, Nt6)

# x7 = np.linspace(0, L, Nx7)
# y7 = np.linspace(0, t_final, Nt7)



# T = np.rot90(T,3)

# print(T3)
# print()
# print(T23)
# # Generate heatmap using seaborn
# plt.figure()
# ax = sns.heatmap(T, cmap='hot', annot=True, fmt='.1f', xticklabels=x, yticklabels=np.arange(0, 61, 20), cbar_kws={'label': 'Temperature (°C)'})
# ax.set_xlabel('x (cm)')
# ax.set_ylabel('Time (s)')
# ax.set_title('Heatmap of temperature distribution along the rod')
# plt.show()
# print(T1)
# print(T6)
# print(T23)
# print((T3[2][1]-T3[2][0])/20)
# print((T23[2][1]-T23[2][0])/20)
print(T3)
plt.plot(y3, T3[1], marker = 'o', label="uneven")
plt.plot(y23[0:7], T23[1][0:7], marker = 'x', label="even")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("Evolution of temperature over time at x = 20 cm")
plt.show()

# plt.plot(y, T1[1], marker = 'o', label="dx = 20 cm, dt = 20 s")
# plt.plot(y3, T3[1], marker = '2', label="dx = 20 cm, dt = 10 s")
# plt.plot(y6, T6[2], marker = 'x', label="dx = 10 cm, dt = 5 s")
# plt.plot(y4, T4[2], marker = '3', label="dx = 10 cm, dt = 10 s")
# plt.plot(y2, T2[2], marker = '1', label="dx = 10 cm, dt = 20 s")
# plt.plot(y5, T5[4], marker = '4', label="dx = 5 cm, dt = 10 s")
# plt.plot(y7, T7[4], marker = '>', label="dx = 5 cm, dt = 5 s")
# plt.xlabel("Time (s)")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.title("Evolution of temperature over time at x = 20 cm")
# plt.show()

# # Plot the temperature distribution
# plt.figure()
# plt.plot(x3, T3[:, -1], marker = '+', label='t = 60')
# plt.plot(x3, T3[:, -2], marker = '1', label='t = 50')
# plt.plot(x3, T3[:, -3], marker = '2', label='t = 30')
# plt.plot(x3, T3[:, -4], marker = '3', label='t = 20')
# plt.plot(x3, T3[:, -5], marker = '4', label='t = 10')
# plt.plot(x3, T3[:, -6], marker = 'x', label='t = 0')
# plt.xlabel("x (cm)")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.show()

# # Plot the temperature distribution
# plt.figure()
# plt.plot(x, T[:, -1], marker = '+', label='t = 60')
# plt.plot(x, T[:, -2], marker = '1', label='t = 40')
# plt.plot(x, T[:, -3], marker = '2', label='t = 20')
# plt.plot(x, T[:, -4], marker = '3', label='t = 0')
# plt.xlabel("x (cm)")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.show()
