import numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

# ========== 題目 1 ==========
f1 = lambda x: np.exp(x) * np.sin(4 * x)
a1, b1, h1 = 1, 2, 0.1
x1 = np.arange(a1, b1 + h1, h1)

def trapezoidal(f, x, h):
    if len(x) < 2:
        return 0.0
    return h * (0.5 * f(x[0]) + sum(f(x[1:-1])) + 0.5 * f(x[-1]))

def simpsons(f, x, h):
    n = len(x) - 1
    if n < 2:
        return 0.0
    if n % 2 != 0:
        if len(x) % 2 == 0:
            x = x[:-1]
            n = len(x) - 1
            if n % 2 != 0:
                return np.nan
    return h / 3 * (f(x[0]) + 4 * sum(f(x[1:-1:2])) + 2 * sum(f(x[2:-1:2])) + f(x[-1]))

def midpoint(f, x, h):
    if len(x) < 2:
        return 0.0
    midpoints = (x[:-1] + x[1:]) / 2
    return h * sum(f(midpoints))

result1a = trapezoidal(f1, x1, h1)
result1b = simpsons(f1, x1, h1)
result1c = midpoint(f1, x1, h1)

print("========== 題目 1 ==========")
print(f"1.a Trapezoidal Rule: {result1a:.8f}")
print(f"1.b Simpson’s Method: {result1b:.8f}")
print(f"1.c Midpoint Rule: {result1c:.8f}")

# ========== 題目 2 ==========
f2 = lambda x: x**2 * np.log(x)
a2, b2 = 1, 1.5

def gauss_quad(f, a, b, n):
    x, w = leggauss(n)
    x_mapped = 0.5 * (x + 1) * (b - a) + a
    return 0.5 * (b - a) * np.sum(w * f(x_mapped))

exact2, _ = quad(f2, a2, b2)
gauss_n3 = gauss_quad(f2, a2, b2, 3)
gauss_n4 = gauss_quad(f2, a2, b2, 4)

print("\n========== 題目 2 ==========")
print(f"2. Gaussian n=3 : {gauss_n3:.8f}")
print(f"2. Gaussian n=4 : {gauss_n4:.8f}")
print(f"2. Exact Value  : {exact2:.8f}")
print(f"2. Error n=3    : {abs(gauss_n3 - exact2):.2e}")
print(f"2. Error n=4    : {abs(gauss_n4 - exact2):.2e}")

# ========== 題目 3 ==========
f3 = lambda x, y: 2 * y * np.sin(x) + np.cos(x)**2

def simpsons_double(f, a, b, n, m):
    hx = (b - a) / n
    x = np.linspace(a, b, n + 1)
    result = 0
    for i in range(n + 1):
        xi = x[i]
        yi_a, yi_b = np.sin(xi), np.cos(xi)
        if yi_b < yi_a or m == 0:
            inner = 0
        else:
            hy = (yi_b - yi_a) / m
            y = np.linspace(yi_a, yi_b, m + 1)
            if m % 2 != 0:
                y = y[:-1]
            inner = simpsons(lambda y_: f(xi, y_), y, hy)
        weight = 4 if i % 2 == 1 else 2
        if i == 0 or i == n:
            weight = 1
        result += weight * inner
    return hx / 3 * result

def gauss_double(f, a, b, nx, ny):
    xg, wx = leggauss(nx)
    yg, wy = leggauss(ny)
    result = 0
    for i in range(nx):
        xi = 0.5 * (xg[i] + 1) * (b - a) + a
        wi = wx[i] * 0.5 * (b - a)
        y1, y2 = np.sin(xi), np.cos(xi)
        for j in range(ny):
            yj = 0.5 * (yg[j] + 1) * (y2 - y1) + y1
            wj = wy[j] * 0.5 * (y2 - y1)
            result += wi * wj * f(xi, yj)
    return result

simpson3 = simpsons_double(f3, 0, np.pi/4, 4, 4)
gauss3 = gauss_double(f3, 0, np.pi/4, 3, 3)
exact3, _ = quad(lambda x: quad(lambda y: f3(x, y), np.sin(x), np.cos(x))[0], 0, np.pi / 4)

print("\n========== 題目 3 ==========")
print(f"3.a Simpson 2D (n=4, m=4): {simpson3:.8f}")
print(f"3.b Gaussian 2D (n=3, m=3): {gauss3:.8f}")
print(f"3.c Exact Value            : {exact3:.8f}")
print(f"3.c Simpson Error          : {abs(simpson3 - exact3):.2e}")
print(f"3.c Gauss Error            : {abs(gauss3 - exact3):.2e}")

# ========== 題目 4 ==========
def composite_simpson_q4(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("子區間數 n 必須是偶數")
    if n == 0:
        return 0.0
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    integral = f(x[0]) + f(x[n])
    for i in range(1, n, 2):
        integral += 4 * f(x[i])
    for i in range(2, n, 2):
        integral += 2 * f(x[i])
    return integral * h / 3

def f_a(x):
    if x == 0:
        return 0.0
    if x < 0:
        return np.nan
    return x**(-0.25) * np.sin(x)

def f_b_transformed(t):
    if t == 0:
        return 0.0
    return t**2 * np.sin(1/t)

n_q4 = 4
result4a = composite_simpson_q4(f_a, 0, 1, n_q4)
result4b = composite_simpson_q4(f_b_transformed, 0, 1, n_q4)

print("\n========== 題目 4 ==========")
print(f"4.a Improper Simpson x^(-1/4): {result4a:.8f}")
print(f"4.b Improper Simpson (∞→1 by t=1/x): {result4b:.8f}")
