import numpy as np
import matplotlib.pylab as plt
from enum import Enum, auto

class Method(Enum):
	BEn = auto()
	BEf = auto()
	TR = auto() 
	BDF4 = auto()
	FE = auto()
	RK4 = auto()

def fn (t,y):
    return np.sin(t * y)

def fn_dy (t,y):
	return t * np.cos(t * y)

def logistisch (t,y):
	return y * (1 - y)

def logistisch_dy (t,y):
	return (1 - y) - y

# Backward Euler with Newton
def bEuler_newton (f, fdy, x0, y0, h, n):
    r = np.empty([n, 1])
    r[0] = y0
    t = x0
    its = 0
    for i in range(1,n):
        yk = r[i-1]
        t += h
        # initial estimate via forward euler
        c = yk + h * f(t,yk)
        n = c - ((yk + h * f(t,c) - c) / (h * fdy(t,c) - 1))
        while abs(c-n) >= 1.0e-10:
            its += 1
            c = n
            n = c - ((yk + h * f(t,c) - c) / (h * fdy(t,c) - 1))
        r[i] = n
    print(its)
    return r

# Backward Euler fixed-point 
def bEuler_fxpt(f, x0, y0, h, n):
    r = np.empty([n, 1])
    r[0] = y0
    t = x0
    its = 0
    for i in range(1,n):
        yk = r[i-1]
        t += h
        c = yk
        n = yk + h * f(t,yk)
        while abs(c-n) >= 1.0e-10:
            its += 1
            c = n
            n = yk + h * f(t,c)
        r[i] = c
    print(its)
    return r

# Trapez
def trapez(f, fdy, x0, y0, h, n):
    r = np.empty([n, 1])
    r[0] = y0
    t = x0
    its = 0
    for i in range(1,n):
        yk = r[i-1]
        t += h
        # Forward euler
        c = yk + h * f(t,yk)
        n = c - ((yk + h * f(t,c) - c) / (h * fdy(t,c) - 1))
        while abs(c-n) >= 1.0e-10:
            its += 1
            c = n
            n = c - ((yk + h * f(t,c) - c) / (h * fdy(t,c) - 1))
        r[i] = n
    print(its)
    return r



# Forward Euler
def fEuler (f, x0, y0, h, n):
    r = np.empty([n, 1])
    r[0] = y0
    t = x0
    for i in range(1,n):
        yk = r[i-1]
        r[i] = yk + h * f(t,yk)
        t += h
    return r


def solve (f, fdy, x0, y0, h, n, m):
    if m == Method.BEn:
        return bEuler_newton(f, fdy, x0, y0, h, n)
    elif m == Method.BEf:
        return bEuler_fxpt(f, x0, y0, h, n)
    elif m == Method.TR:
        return trapez(f, fdy, x0, y0, h, n)
    elif m == Method.FE:
        return fEuler(f, x0, y0, h, n)
    else:
        print("Solver not yet implemented")
        exit(0)

def plot (f, fdy, x0, y0, xn, h, m):
    x = np.linspace(x0, xn, (xn-x0)/h)
    y = solve(f, fdy, x0, y0, h, int((xn-x0)/h), m)

    plt.plot(x, y)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.axis('tight')
    plt.show()

plot(fn, fn_dy, -20, 10, 20, 0.2, Method.BEn)
#plot(logistisch, logistisch_dy, 0, 0.5, 6, 0.1, Method.BEn)
