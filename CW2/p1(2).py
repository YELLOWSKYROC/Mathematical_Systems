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

# print(Ta)
# print(Ta2)



# Parameters and Initial Conditions
L = 100
t = 600
kappa = 0.835
T_initial = 500

# Time and space discretization
dx = 20
dt = 100
Nx = int(L / dx) + 1
Nt = int(t / dt) + 1

dx2 = 20
dt2 = 50
Nx2 = int(L / dx2) + 1
Nt2 = int(t / dt2) + 1

dx3 = 10
dt3 = 100
Nx3 = int(L / dx3) + 1
Nt3 = int(t / dt3) + 1

# Initialize temperature array
T = np.zeros((Nt, Nx))
T[:, 1:5] = T_initial
x1 = np.linspace(0, L, Nx)

T2 = np.zeros((Nt2, Nx2))
T2[:, 1:5] = T_initial
x2 = np.linspace(0, L, Nx2)

T3 = np.zeros((Nt3, Nx3))
T3[:, 1:10] = T_initial
x3 = np.linspace(0, L, Nx3)


# Explicit finite-difference method
alpha = kappa * dt / dx**2
for n in range(0, Nt - 1):
    for i in range(1, Nx - 1):
        T[n + 1, i] = T[n, i] + alpha * (T[n, i - 1] - 2 * T[n, i] + T[n, i + 1])


alpha2 = kappa * dt2 / dx2**2
for n in range(0, Nt2 - 1):
    for i in range(1, Nx2 - 1):
        T2[n + 1, i] = T2[n, i] + alpha2 * (T2[n, i - 1] - 2 * T2[n, i] + T2[n, i + 1])


alpha3 = kappa * dt3 / dx3**2
for n in range(0, Nt3 - 1):
    for i in range(1, Nx3 - 1):
        T3[n + 1, i] = T3[n, i] + alpha3 * (T3[n, i - 1] - 2 * T3[n, i] + T3[n, i + 1])


e1 = [abs(Ta[i]-list(T[:, 1])[i]) for i in range(0,len(Ta))]
e2 = [abs(Ta2[i]-list(T2[:, 1])[i]) for i in range(0,len(Ta2))]
e3 = [abs(Ta[i]-list(T3[:, 2])[i]) for i in range(0,len(Ta))]


# print(e1)
print(e3)

# a = [1,2,3,4,5]

# b = [5,4,3,2,1]

# c = [a[i]-b[i] for i in range(0,len(a))] #range后还可以加if条件筛选

# print(c)

# plot the evolution of temperature over time at x = 20 cm
plt.figure()
plot1 = plt.plot(np.arange(0, 601, 50), Ta2, marker = 'o', label='Analytical solution')
plt.plot(np.arange(0, 601, 100), T[:, 1], marker = '+', label='label="dx = 20 cm, dt = 100 s"')
plt.plot(np.arange(0, 601, 100), e1, marker = '+', label='absolute error1')
plt.plot(np.arange(0, 601, 50), T2[:, 1], marker = '1', label='label="dx = 20 cm, dt = 50 s"')
plt.plot(np.arange(0, 601, 50), e2, marker = '1', label='absolute error2')

# plt.plot(np.arange(0, 601, 100), T3[:, 2], marker = '2', label='label="dx = 10 cm, dt = 100 s"')
# plt.plot(np.arange(0, 601, 100), e3, marker = '2', label='absolute error')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()


# #  plot the temperature distribution along the bar at 100 s intervals
# plt.plot(x1, T[0, :], marker = 'o', label="T = 0 s")
# plt.plot(x1, T[1, :], marker = '1', label="T = 100 s")
# plt.plot(x1, T[2, :], marker = '2', label="T = 200 s")
# plt.plot(x1, T[3, :], marker = '3', label="T = 300 s")
# plt.plot(x1, T[4, :], marker = '4', label="T = 400 s")
# plt.plot(x1, T[5, :], marker = '+', label="T = 500 s")
# plt.plot(x1, T[6, :], marker = 'x', label="T = 600 s")
# plt.xlabel('x (cm)')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.show()



# # plot the evolution of temperature over time at x = 20 cm
# plt.figure()
# plot1 = plt.plot(np.arange(0, 601, 100), Ta, marker = 'o', label='Analytical solution')
# # plt.plot(np.arange(0, 601, 100), T[:, 1], marker = '+', label='label="dx = 20 cm, dt = 100 s"')
# # plt.plot(np.arange(0, 601, 50), T2[:, 1], marker = '1', label='label="dx = 20 cm, dt = 50 s"')
# plt.plot(np.arange(0, 601, 100), T3[:, 2], marker = '2', label='label="dx = 10 cm, dt = 100 s"')
# plt.xlabel('Time (s)')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.show()


# # Generate heatmap using seaborn
# plt.figure()
# ax = sns.heatmap(T, cmap='hot', annot=True, fmt='.1f', xticklabels=x1, yticklabels=np.arange(0, 601, 100), cbar_kws={'label': 'Temperature (°C)'})
# ax.set_xlabel('x (cm)')
# ax.set_ylabel('Time (s)')
# ax.set_title('Heatmap of temperature distribution along the rod')
# plt.show()

# plot the temperature distribution along the bar at 100 s intervals in T3.
# plt.figure()
# # plt.plot(x3, T3[0, :], label='t = 0 s')
# # plt.plot(x3, T3[1, :], label='t = 100 s')
# # plt.plot(x3, T3[2, :], label='t = 200 s')
# # plt.plot(x3, T3[3, :], label='t = 300 s')
# # plt.plot(x3, T3[4, :], label='t = 400 s')
# # plt.plot(x3, T3[5, :], label='t = 500 s')
# plt.plot(x3, T3[6, :], label='t = 600 s')
# plt.xlabel('x (cm)')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.show()
