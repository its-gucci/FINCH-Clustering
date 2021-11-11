import numpy as np


def clamp(a, eps=1e-12):
    """
    Makes things non-negative
    """
    return np.maximum((a - eps), 0) + eps

def artanh(x):
    x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    return 0.5 * (np.log(1 + x) - np.log(1 - x))

def arcosh(x):
    c = clamp(x**2 - 1)
    return np.log(clamp(x + np.sqrt(c)))

def np_mobius_add(x, y, c, dim=-1):
    y = y + 1e-15
    x2 = np.sum(np.power(x, 2), axis=dim, keepdims=True)
    y2 = np.sum(np.power(y, 2), axis=dim, keepdims=True)
    xy = np.sum(x * y, axis=dim, keepdims=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-15)

def np_dist(x, y, c=1.0, keepdim=False, dim=-1):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * np.linalg.norm(np_mobius_add(-x, y, c, dim=dim), axis=dim, keepdims=keepdim))
    return dist_c * 2 / sqrt_c

def g(x):
    return (2 * arcosh(1 + 2 * x))/clamp(np.sqrt(x**2 + x))

def frechet_mean(x, iterations=5):
    yk = x[0]
    for k in range(iterations):
        a = 0
        b = 0
        c = 0
        for l in range(len(x)):
            num  = np.inner(x[l] - yk, x[l] - yk)
            denom1 = 1 - np.inner(x[l], x[l])
            denom2 = 1 - np.inner(yk, yk)
            alpha = g(num/(denom1 * denom2))/denom1
            a += alpha
            b += alpha * x[l]
            c += alpha * np.inner(x[l], x[l])
        b2 = np.inner(b, b)
        yk = ((a + c - np.sqrt(clamp((a + c)**2 - 4 * b2)))/(2 * b2)) * b
    return yk