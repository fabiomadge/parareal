import numpy as np
import math as m
import matplotlib.pylab as plt
from enum import Enum, auto

MAXITER = 100
NEWTON_ACC = 1.0e-12

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

def newton(f, fdx, x0):
    itr = 1
    c = x0
    # Lokales Extremum: Schlecht
    n = c - f(c) / fdx(c)
    # Sollte vermutlich relativ zu x sein.
    while abs(c-n) >= NEWTON_ACC:
        #print(abs(c-n))
        if itr >= MAXITER:
            return None
        itr += 1
        c = n
        n = c - f(c) / fdx(c)
    return (n, itr)


# Backward Euler with Newton
def bEuler_newton (f, fdy, x0, y0, h, n):
    r = np.zeros(n)
    r[0] = y0
    t = x0
    its = 0
    for i in range(1,n):
        yk = r[i-1]
        t += h
        step = lambda ykpp: yk + h * f(t,ykpp) - ykpp
        stepDx = lambda ykpp: h * fdy(t,ykpp) - 1
        # forward euler
        estimate = yk + h * f(t,yk)
        res = newton(step, stepDx, estimate)
        if res == None:
            print("MaxIter exeeded at t=%f" % t)
            return None
        r[i] = res[0]
        its += res[1]
    print("%d iteration over %d steps" % (its, n))
    return r

# Backward Euler fixed-point 
def bEuler_fxpt(f, x0, y0, h, n):
    r = np.zeros(n)
    r[0] = y0
    t = x0
    its = 0
    for i in range(1,n):
        itr = 0
        yk = r[i-1]
        t += h
        c = yk
        x = yk + h * f(t,yk)
        while not m.isinf(x) and abs(c-x) >= 1.0e-10:
            if itr >= MAXITER:
                print("MaxIter exeeded at t=%f" % t)
                return r
            itr += 1
            c = x
            x = yk + h * f(t,c)
        if m.isinf(x):
            print("fixed-point iteration diverged")
            return None
        r[i] = x
        its += itr
    print("%d iteration over %d steps" % (its, n))
    return r

# Trapez with Newton iteration
def trapez(f, fdy, x0, y0, h, n):
    r = np.zeros(n)
    r[0] = y0
    t = x0
    its = 0
    for i in range(1,n):
        yk = r[i-1]
        t += h
        dyk = f(t,yk)
        step = lambda ykpp: yk + .5 * h * (dyk + f(t,ykpp)) - ykpp
        stepDx = lambda ykpp: .5 * h * fdy(t,ykpp) - 1
        # forward euler
        estimate = yk + h * f(t,yk)
        res = newton(step, stepDx, estimate)
        if res == None:
            print("MaxIter exeeded at t=%f" % t)
            return None
        r[i] = res[0]
        its += res[1]
    print("%d iteration over %d steps" % (its, n))
    return r

# BDF with Newton iteration and maxord
def bdf(f, fdy, x0, y0, h, n, maxord):
    r = np.zeros(n)
    r[0] = y0
    t = x0
    its = 0
    for i in range(1,n):
        order = min(i,maxord)
        t += h
        if order == 1:
            step = lambda yi: r[i-1] + h * f(t,yi) - yi
            stepDx = lambda yi: h * fdy(t,yi) - 1
        elif order == 2:
            step = lambda yi: yi - 4/3 * r[i-1] + 1/3 * r[i-2] - 2/3 * h * f(t,yi)
            stepDx = lambda yi: 1 - 2/3 * h * fdy(t,yi)
        elif order == 3:
            step = lambda yi: yi - 18/11 * r[i-1] + 9/11 * r[i-2] - 2/11 * r[i-3] - 6/11 * h * f(t,yi)
            stepDx = lambda yi: 1 - 6/11 * h * fdy(t,yi)
        elif order == 4:
            step = lambda yi: yi - 48/25 * r[i-1] + 36/25 * r[i-2] - 16/25 * r[i-3] + 3/25 * r[i-4]- 12/25 * h * f(t,yi)
            stepDx = lambda yi: 1 - 12/25 * h * fdy(t,yi)
        elif order == 5:
            step = lambda yi: yi - 300/137 * r[i-1] + 300/137 * r[i-2] - 200/137 * r[i-3] + 75/137 * r[i-4] - 12/137 * r[i-5] - 60/137 * h * f(t,yi)
            stepDx = lambda yi: 1 - 60/137 * h * fdy(t,yi)
        elif order == 6:
            step = lambda yi: yi - 360/147 * r[i-1] + 450/147 * r[i-2] - 400/147 * r[i-3] + 225/147 * r[i-4] - 72/147 * r[i-5] + 10/147 * r[i-6] - 60/147 * h * f(t,yi)
            stepDx = lambda yi: 1 - 60/147 * h * fdy(t,yi)
        else:
            print("order=%d is unsupported" % order)
            exit(0)
        # forward euler
        estimate = r[i-1] + h * f(t,r[i-1])
        res = newton(step, stepDx, estimate)
        if res == None:
            print("MaxIter exeeded at t=%f" % t)
            return None
        r[i] = res[0]
        its += res[1]
    print("%d iteration over %d steps" % (its, n))
    return r

# Forward Euler
def fEuler (f, x0, y0, h, n):
    r = np.zeros(n)
    r[0] = y0
    t = x0
    for i in range(1,n):
        yk = r[i-1]
        r[i] = yk + h * f(t,yk)
        t += h
    return r

def rk4 (f, x0, y0, h, n):
    r = np.zeros(n)
    r[0] = y0
    t = x0
    for i in range(1,n):
        yk = r[i-1]
        k1 = f(t,yk)
        k2 = f(t + h/2, yk + h*k1/2)
        k3 = f(t + h/2, yk + h*k2/2)
        k4 = f(t + h,   yk + h*k3)
        r[i] = yk + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return r


def solve (f, fdy, x0, y0, h, n, m):
    if m is Method.BEn:
        return bEuler_newton(f, fdy, x0, y0, h, n)
    elif m is Method.BEf:
        return bEuler_fxpt(f, x0, y0, h, n)
    elif m is Method.TR:
        return trapez(f, fdy, x0, y0, h, n)
    elif m is Method.BDF4:
        return bdf(f, fdy, x0, y0, h, n, 4)
    elif m is Method.FE:
        return fEuler(f, x0, y0, h, n)
    elif m is Method.RK4:
        return rk4(f, x0, y0, h, n)
    else:
        print("Solver %s not yet implemented" % m)
        exit(0)

def plot (f, fdy, x0, y0, xn, h, m):
    x = np.linspace(x0, xn, int((xn-x0)/h)+1)
    y = solve(f, fdy, x0, y0, h, int((xn-x0)/h)+1, m)

    # print(y)

    if not (y is None):
        plt.plot(x, y)
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.axis('tight')
        plt.show()

def error (f, fdy, x0, y0, xn, maxExp, sol):
    res = np.zeros([len(Method), maxExp+1])
    if sol is None:
        intervals = 2**(maxExp+4)
        sol = solve(f, fdy, x0, y0, (xn-x0)/intervals, intervals+1, Method.RK4)[0::int(2**4)]
    else:
        sol = sol(np.linspace(x0, xn, 2**maxExp+1))
    for md in Method:
        print(md)
        for i in range(0,maxExp+1):
            intervals = 2**i
            apx = solve(f, fdy, x0, y0, (xn-x0)/intervals, intervals+1, md)
            if apx is None:
                res[md.value-1][i] = 0
            else:
                #maximumsnorm
                res[md.value-1][i] = np.max(np.abs(apx - (sol[0::2**(maxExp-i)])))
                #euklidische norm
                #res[md.value-1][i] = m.sqrt(np.sum((apx - (sol[0::2**(maxExp-i)]))**2)/(intervals+1))
                #summennorm
                #res[md.value-1][i] = np.sum(np.abs(apx - (sol[0::2**(maxExp-i)])))/(intervals+1
    plt.loglog(2**np.linspace(0,maxExp,maxExp+1), np.transpose(res))
    plt.legend(Method)
    plt.show()
    return None

def parareal(f, fdy, x0, y0, xn, exp, procExp, coarse, fine):
    procs = 2**procExp
    hC = (xn-x0)/procs
    u = solve(f, fdy, x0, y0, hC, procs+1, coarse)
    r = np.zeros((2**exp)+1)
    fIpP = 2**(exp-procExp)
    kMAX = 4
    for k in range(0,kMAX):
        d = np.zeros(procs+1)
        for j in range(1,procs+1):
            g = solve(f, fdy, x0+(j*hC), u[j-1], hC, 2, coarse)
            fn = solve(f, fdy, x0+(j*hC), u[j-1], hC/fIpP, fIpP+1, fine)
            if k == kMAX-1:
                r[fIpP*(j-1):fIpP*j+1] = fn
            d[j] = fn[fIpP] - g[1]
        cc = np.zeros(procs+1)
        for j in range(1,procs+1):
            g = solve(f, fdy, x0+(j*hC), u[j-1], hC, 2, coarse)
            cc[j] = g[1]
            u[j] = g[1] + d[j]
    return (u, r)

def plot_pr (f, fdy, x0, y0, xn, exp, procExp, coarse, fine):
    xr = np.linspace(x0, xn, (2**exp)+1)
    xu = np.linspace(x0, xn, (2**procExp)+1)
    (u,r) = parareal(f, fdy, x0, y0, xn, exp, procExp, coarse, fine)

    if not (u is None):
        plt.plot(xr, r)
        plt.plot(xu, u)
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.axis('tight')
        plt.show()

#plot(fn, fn_dy, -20, 10, 20, 0.001, Method.BDF4)
#plot(logistisch, logistisch_dy, -6, 1/(1+m.e**6), 6, 12/2**4, Method.BEn)
#print(newton(lambda x: (x-1)*(x+1), lambda x: (x+1)+(x-1), 0.000001))
#error(logistisch, logistisch_dy, -6, 1/(1+m.e**6), 6, 14, None)
#error(fn, fn_dy, -20, 10, 20, 18, None)
#error(logistisch, logistisch_dy, -6, 1/(1+m.e**6), 6, 18, lambda x: 1/(1+(m.e**(-x))))
plot_pr(logistisch, logistisch_dy, -6, 1/(1+m.e**6), 6, 8, 3, Method.FE, Method.RK4)
