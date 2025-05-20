import numpy as np
import pandas as pd
from numba import jit
import time as tm
from mpi4py import MPI

st = tm.time()

pd.set_option('display.max_rows', 1000000)

m = 100
dt = 0.01
t0 = 0
time = 1000
Dm_val = np.arange(0.05, 0.85, 0.05)
Dp_val = [0.2, 0.4, 0.6, 0.8]

@jit(nopython = True)
def func(x0, Dminus, Dplus, y):
    def noise(x, Dminus, Dplus, dt):
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

    def z(x, t):
        a = 1
        b = 1
        ans = a*x - b*x**3
        return ans

    def single(z, x0, t0, tf, dt, y):
        num_steps = int((tf - t0) / dt)
        x_val = []
        mfpt = 0
        for _ in range(1, num_steps):
            k1 = dt * z(x0, t0)
            k2 = dt * z(x0 + 0.5*k1, t0 + 0.5*dt)
            x1 = x0 + 0.5 * (k1 + k2) + noise(x0, Dminus, Dplus, dt)
            if (x1 >= y):
                mfpt = t0
                break
            t0 = t0 + dt
            x0 = x1
            x_val.append(x0)
        return mfpt

    ans = []
    for _ in range(m):
        a = single(z, x0, t0, time, dt, y)
        if a!= 0:
            ans.append(a)
    average_time = sum(ans)/len(ans)
    return 1/average_time

def traj_avg(Dminus, Dplus, y):
    a = []
    x0_val = np.linspace(-1.2, -0.5, 100)
    for x0 in x0_val:
        a.append(func(x0, Dminus, Dplus, y))
    avg = sum(a)/len(a)
    return avg

def main(Dplus, y):
    print(f"Calculation running for Dplus = {Dp}")
    rate = []
    for Dm in Dm_val:
        ans = traj_avg(Dm, Dplus, y)
        rate.append(ans)
    return rate

y = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Divide the array among processes
Dp_split = np.array_split(Dp_val, size)[rank]

# Perform calculations for each number in the local array
results = []
for Dp in Dp_split:
    result = main(Dp, y)  # Call your function func() here
    results.append(result)

# Gather results from all processes to process 0
gathered_results = comm.gather(results, root=0)

# Combine gathered results into a single array on process 0
if rank == 0:
    rate = np.concatenate(gathered_results)
print((str(pd.Series(Dp_val, rate))))

et = tm.time()
print(f'Time required is: {et-st} sec')

