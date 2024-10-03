# Analytical Solution of One-dimensional heat conduction equation
import numpy as np
import matplotlib.pyplot as plt

# Analytical solution function
def analytical_solution(x, t):
    temperature = 0
    for n in range(1, 1000, 2):
        temperature += (1/n)*np.sin((n*np.pi*x/100))*np.exp((-0.835*(n*np.pi/100)**2)*t)
    return (2000/np.pi)*temperature

temperature = analytical_solution(20, 100)
print(temperature)

T = []
for i in range(0, 601, 100):
    T.append(analytical_solution(20, i))

print(T)
t = [0, 100, 200, 300, 400, 500, 600]

# # Calculate and plot the solution
# for time in t:
#     temperature = analytical_solution(x, time, L, T0, alpha)

plt.plot(t, T)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (T)')
plt.title('Analytical Solution of One-dimensional Heat Conduction Equation')
plt.show()