import numpy as np
import matplotlib.pylab as plt
from enum import Enum, auto

class Method(Enum):
	BE = auto()
	TR = auto() 
	BDF4 = auto()
	FE = auto()
	RK4 = auto()

def fn (t,y):
    return np.sin(t * y)

def logistisch (t,y):
	return y * (1 - y)

# Backwards Euler
def bEuler (f, x0, y0, h, n):
    r = np.empty([n, 1])
    r[0] = y0
    t = x0
    for i in range(1,n):
        yk = r[i-1]
        t += h
        c = yk
        n = yk + h * f(t,yk)
        while abs(c-n) >= 1.0e-10:
            c = n
            n = yk + h * f(t,c)
        r[i] = c
    return r

def solve (f, x0, y0, h, n, m):
	if m == Method.BE:
		return bEuler(f, x0, y0, h, n)

def plot (f, x0, y0, xn, h, m):
    x = np.linspace(x0, xn, (xn-x0)/h)
    y = solve(f, x0, y0, h, int((xn-x0)/h), m)

    plt.plot(x, y)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.axis('tight')
    plt.show()

#plot(fn, -20, 10, 20, 0.01)
plot(logistisch, 0, 0.5, 6, 0.01, Method.BE)
