from mpi4py import MPI
import numpy as np
import pandas as pd
import time as tm
import numba as nb

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

st = tm.time()

pd.set_option('display.max_rows', 1000000)

m = 500
dt = 0.005
t0 = 0
time = 1000
Dplus = 0.08

@nb.jit(nopython=True)
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

@nb.jit(nopython=True)
def z(x, t):
    a = 1
    b = 1
    ans = a*x - b*x**3
    return ans

@nb.jit(nopython=True)
def single(x0, Dplus, Dminus, y):
    def single_trajectory(z, x0, t0, tf, dt, y):
        num_steps = int((tf - t0) / dt)
        mfpt = 0
        for _ in range(1, num_steps):
            k1 = dt * z(x0, t0)
            k2 = dt * z(x0 + 0.5*k1, t0 + 0.5*dt)
            x1 = x0 + 0.5 * (k1 + k2) + noise(x0, Dplus, Dminus, dt)
            if (x1 >= y):
                mfpt = t0
                break
            t0 = t0 + dt
            x0 = x1
        return mfpt

    ans = []
    for _ in range(m):
        a = single_trajectory(z, x0, t0, time, dt, y)
        if a != 0:
            ans.append(a)
    if len(ans) == 0:
        return 0
    average_time = sum(ans)/len(ans)
    return 1/average_time

@nb.jit(nopython=True)
def traj_avg(Dplus, Dminus, y):
    x0_val = np.linspace(-1.1, -0.9, 500)
    results = [single(x0, Dplus, Dminus, y) for x0 in x0_val]
    results = [res for res in results if res != 0]  # Filter out zero rates
    if len(results) == 0:
        return 0
    avg = sum(results)/len(results)
    return avg

Dm_val = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13]
num_Dm_vals = len(Dm_val)

if rank < num_Dm_vals:
    # Each processor handles one y_val
    i = Dm_val[rank]
    ans = []
    y_val = np.arange(0, 1.1, 0.1)
    for y in y_val:
        a = traj_avg(Dplus, i, y)
        print(f"Processor {rank}: The Rate is: {a:.5f} at reference point being {y:.2f} for Dminus at {i} and Dplus at {Dplus}")
        ans.append(a)

    # Save D_val and ans to a text file for each y_val
    output_filename = f"results_Dm_{i:.2f}.txt"
    with open(output_filename, "w") as file:
        file.write("y_val\tRate\n")
        for y, a in zip(y_val, ans):
            file.write(f"{y:.2f}\t{a}\n")

    # Gather all results to rank 0 for plotting
    all_y_vals = comm.gather(y_val, root=0)
    all_ans_vals = comm.gather(ans, root=0)
else:
    all_y_vals = None
    all_ans_vals = None

#print(str(pd.Series(all_ans_vals, all_Dm_vals)))

et = tm.time()
print(f'Time required is: {et-st} sec')

# Finalize MPI
MPI.Finalize()
