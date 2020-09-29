import numpy as np


def fer_MC(x, y):
    n_trails, n = y.shape
    fer = 0
    res = []
    for i in range(n_trails):
        if abs(x[i, :] - y[i, :]).sum() != 0:
            fer = fer + 1 / n_trails
            res.append(1)
        else:
            res.append(0)
    std = np.std(res)
    cv = std / (fer + 1e-15)    # coefficient of variation
    return fer, cv


def fer_IS(x, y, weights):
    n_trails, n = y.shape
    fer = 0
    res = []
    for i in range(n_trails):
        if abs(x[i, :] - y[i, :]).sum() != 0:
            fer = fer + 1*weights[i] / n_trails
            res.append(weights[i])
        else:
            res.append(0)
    std = np.std(res)
    cv = std / (fer + 1e-15)    # coefficient of variation
    return fer, cv


def ber_IS(x, y, weights):
    n_trials, n = y.shape
    ber = 0
    for i in range(n_trials):
        ber += abs(x[i, :] - y[i, :]).sum() * weights[i] / (n_trials * n)
    return ber


def ber_MC(x, y):
    n_trials, n = y.shape
    ber = 0
    for i in range(n_trials):
        ber += abs(x[i,:] - y[i,:]).sum() / (n_trials * n)
    return ber

