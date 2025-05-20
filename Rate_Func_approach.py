import numpy as np
import matplotlib.pyplot as plt

def rate(x):
    Dplus = 0.05
    Dminus = 0.5
    C1 = 0.004
    C2 = 0.005
    def U(x):
        return 0.5*x**2 - 0.25*x**4

    def D(x, Dplus, Dminus):
        if x < 0:
            return Dplus
        elif x > 0:
            return Dminus
        else:
            return 0

    a = np.exp(-U(x)/D(x, Dplus, Dminus))
    b = D(x, Dplus, Dminus)
    c = x - x**3
    return C1*b*a/c - x + C2

i = -0.99
a = []
x_val = []
while(i<1):
    ans = rate(i)
    a.append(ans)
    x_val.append(i)
    i = i + 0.01

plt.plot(x_val, a)
plt.show()