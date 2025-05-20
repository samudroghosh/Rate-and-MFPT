import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
import time as tm
import ray

ray.init(num_cpus = 12)

st = tm.time()

pd.set_option('display.max_rows', 1000000)

m = 1000
dt = 0.01
t0 = 0
time = 1000
omega_val = np.arange(0, 101, 5)
Dplus = 0.4
Dminus = 0.3

def func(x0, Dplus, Dminus, omega):
    def noise(x, Dplus, Dminus, dt):
        if x < 0:
            D = Dminus
        elif x > 0:
            D = Dplus
        else:
            D = 0
        U1 = np.random.uniform(0, 1, size=1)
        U2 = np.random.uniform(0, 1, size=1)
        R = -4*D*dt*np.log(U1)
        Theta = 2*np.pi*U2
        X = np.sqrt(R) * np.sin(Theta)
        return X[0]

    def z(x, t, omega):
        omu = np.radians(omega)
        driving = 0.1*np.cos(omu*t)
        return x - x**3 + driving

    def single(z, x0, t0, tf, dt, omega):
        num_steps = int((tf - t0) / dt)
        t_values = np.linspace(t0, tf, num_steps + 1)
        x_values = np.zeros(len(t_values))
        x_values[0] = x0
        mfpt = 0
        for i in range(1, len(t_values)):
            k1 = dt * z(t_values[i-1], x_values[i-1], omega)
            k2 = dt * z(t_values[i-1] + dt, x_values[i-1] + k1, omega)
            x_values[i] = x_values[i-1] + 0.5 * (k1 + k2) + noise(x_values[i-1],Dplus, Dminus, dt)
            if (x_values[i] < 0):
                mfpt = t_values[i]
                break
        return mfpt

    ans = []

    for _ in range(m):
        a = single(z, x0, t0, time, dt, omega)
        if a!= 0:
            ans.append(a)
    average_time = sum(ans)/len(ans)
    return 1/average_time

@ray.remote
def traj_avg(Dplus, Dminus, omega):
    a = []
    x0_val = np.linspace(1.1, 2.1, 100)
    for x0 in x0_val:
        a.append(func(x0, Dplus, Dminus, omega))
    avg = sum(a)/len(a)
    print("The rate is: ",avg, "for omega: ", omega)
    return avg

result = []
result = [traj_avg.remote(Dplus, Dminus, omega) for omega in omega_val]
a = ray.get(result)

#print(str(pd.Series(D, rate)))

et = tm.time()
print(f'Time required is: {et-st} sec')

plt.plot(omega_val, a)
plt.xlabel("Omega")
plt.ylabel("MFPT")
plt.show()