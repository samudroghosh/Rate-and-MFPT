import numpy as np
import pandas as pd
from numba import jit
import time as tm
from mpi4py import MPI

st = tm.time()

pd.set_option('display.max_rows', 1000000)

m = 1000
dt = 0.01
t0 = 0
time = 10000
omega_val = np.arange(0, 361, 1)
D = 0.7
y = 2.1

@jit(nopython = True)
def func(x0, D, omega, y):
    def noise(x, D, dt):
        U1 = np.random.uniform(0, 1, size=1)
        U2 = np.random.uniform(0, 1, size=1)
        R = -4*D*dt*np.log(U1)
        Theta = 2*np.pi*U2
        X = np.sqrt(R) * np.sin(Theta)
        return X[0]

    def z(x, t, y, omega):
        omu = np.radians(omega)
        driving = 0.1*np.sin(omu*t)
        if x <= np.pi:
            ans = 0
        elif x >= 6*np.pi:
            ans = 0
        else:
            ans = 0.5*y*np.sin(y*x) + driving
        return ans

    def single(z, x0, t0, tf, dt, omega, y):
        num_steps = int((tf - t0) / dt)
        t_values = np.linspace(t0, tf, num_steps + 1)
        x_values = np.zeros(len(t_values))
        x_values[0] = x0
        mfpt = 0
        for i in range(1, len(t_values)):
            k1 = dt * z(t_values[i-1], x_values[i-1], y, omega)
            k2 = dt * z(t_values[i-1] + dt, x_values[i-1] + k1, y, omega)
            x_values[i] = x_values[i-1] + 0.5 * (k1 + k2) + noise(x_values[i-1], D, dt)
            if (x_values[i] <= np.pi):
                t_values[i] = t_values[i-1]
            if (x_values[i] > 12.1):
                mfpt = t_values[i]
                break
        return mfpt

    ans = []

    for _ in range(m):
        a = single(z, x0, t0, time, dt, omega, y)
        if a!= 0:
            ans.append(a)
    average_time = sum(ans)/len(ans)
    return 1/average_time

def traj_avg(D, omega, y):
    a = []
    print("Calculation running for: ", omega)
    x0_val = np.linspace(4.0, 5.0, 100)
    for x0 in x0_val:
        a.append(func(x0, D, omega, y))
    avg = sum(a)/len(a)
    return avg

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Divide the array among processes
omega_split = np.array_split(omega_val, size)[rank]

# Perform calculations for each number in the local array
results = []
for omega in omega_split:
    result = traj_avg(D, omega, y)  # Call your function func() here
    results.append(result)

# Gather results from all processes to process 0
gathered_results = comm.gather(results, root=0)

# Combine gathered results into a single array on process 0
if rank == 0:
    rate = np.concatenate(gathered_results)
print((str(pd.Series(omega_val, rate))))

et = tm.time()
print(f'Time required is: {et-st} sec')
