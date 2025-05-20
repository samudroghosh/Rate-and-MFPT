import numpy as np
import matplotlib.pyplot as plt

def mfpt(Dplus, Dminus):
    def F(x):
        return x - x**3

    def D(x, Dplus, Dminus):
        if x > 0:
            return Dplus
        elif x < 0:
            return Dminus
        else:
            return 0

    # Define the function f(x, y1, y2) for the system of first-order differential equations
    def f(x, y1, y2):
        return y2, (F(x)/D(x, Dplus, Dminus)) * y2 + 1/D(x, Dplus, Dminus)

    # RK4 method for solving a system of first-order ODEs
    def rk4_step(x, y1, y2, h):
        k1_y1, k1_y2 = f(x, y1, y2)
        k2_y1, k2_y2 = f(x + 0.5 * h, y1 + 0.5 * h * k1_y1, y2 + 0.5 * h * k1_y2)
        k3_y1, k3_y2 = f(x + 0.5 * h, y1 + 0.5 * h * k2_y1, y2 + 0.5 * h * k2_y2)
        k4_y1, k4_y2 = f(x + h, y1 + h * k3_y1, y2 + h * k3_y2)
        
        y1_next = y1 + (h / 6.0) * (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1)
        y2_next = y2 + (h / 6.0) * (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2)
        
        return y1_next, y2_next

    # Initial conditions
    x0 = -1.0    # Initial value of x
    y1_0 = 0.0  # Initial value of y(x0)
    y2_0 = 1.0  # Initial value of dy/dx(x0)

    # Integration parameters
    h = 0.01  # Step size
    x_end = 1.0  # End value of x

    # Arrays to store the solution
    x_vals = np.arange(x0, x_end, h)
    y1_vals = np.zeros_like(x_vals)
    y2_vals = np.zeros_like(x_vals)

    # Set initial values
    y1_vals[0] = y1_0
    y2_vals[0] = y2_0

    # Perform RK4 integration
    for i in range(1, len(x_vals)):
        y1_vals[i], y2_vals[i] = rk4_step(x_vals[i-1], y1_vals[i-1], y2_vals[i-1], h)

    mfpt = y2_vals[np.where(x_vals >= 0)]
    print(f"The MFPT is: {mfpt[0]}")
    return mfpt[0], y2_vals, x_vals

Dplus = 0.8
Dm_val = [0.2, 0.4, 0.6, 0.8, 1.0]
a = []
for i in Dm_val:
    ans, y, x = mfpt(Dplus, i)
    a.append(ans)
    plt.plot(x, y, label=f'Dminus = {i}')

# Plot the results

#plt.plot(Dm_val, a, label='Dp = 0.05')
plt.xlabel('Position (x)')
plt.ylabel('MFPT')
plt.legend()
plt.title("Dplus = 0.8")
#plt.grid(True)
plt.show()
