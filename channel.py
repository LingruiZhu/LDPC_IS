import numpy.random as rnd
from scipy import stats
import numpy as np
from datetime import datetime
from commpy.utilities import signal_power


def awgn_MC(x, snr, return_noise = False):
    now = datetime.now()
    rng = rnd.default_rng(now.microsecond)
    linear_snr = 10 ** (snr / 20)

    sig_power = signal_power(x.flatten())
    noise_sigma = sig_power / linear_snr

    mu, sigma = 0, noise_sigma
    noise = rng.normal(mu, sigma, x.shape)
    y = x + noise
    if return_noise:
        return y, return_noise
    else:
        return y


def awgn_IS(x, snr, bias_bits, mode='VS'):
    now = datetime.now()
    rng = rnd.default_rng(now.microsecond)
    sig_power = signal_power(x.flatten())
    linear_snr = 10 ** (snr / 10)
    noise_sigma = sig_power / linear_snr

    n_trials, n = x.shape

    if mode == 'VS':
        mu, sigma = 0, noise_sigma
        mu_biased, sigma_biased = 0, 1.8*noise_sigma
    elif mode == 'MT':
        mu, sigma = 0, noise_sigma
        mu_biased, sigma_biased = -0.5, noise_sigma


    # sample noise (from original pdf)
    noise = rng.normal(mu, sigma, x.shape)

    for i in range(len(bias_bits)):
        for j in range(n_trials):
            noise[j, i] = rng.normal(mu_biased, sigma_biased, 1)

    weights = (-1) * np.ones([n_trials])
    for j in range(n_trials):
        weight_temp = 1
        for i in range(len(bias_bits)):
            lr = stats.norm.pdf(noise[j, i], mu, sigma) / stats.norm.pdf(noise[j, i], mu_biased, sigma_biased)
            weight_temp = weight_temp * lr
        weights[j] = weight_temp

    y = x + noise
    return y, weights


def awgn_MIS(x, snr, bias_bits, epsilon, mode = 'MT'):
    """
    Add noise by a multiple gaussian distribution
    :param epsilon: biased value
    :param x: input vector of channel
    :param snr: signal to noise ratio
    :param bias_bits: list. bias bits (trapping sets) are elements of list
    :param mode: MT - Mean transition or VS - variance scaling
    :return: output vector of channel
    """

    # set random generator and seed
    sig_power = signal_power(x.flatten())
    mu = 0
    linear_snr = 10 ** (snr / 10)
    sigma = sig_power / linear_snr
    mu_biased = epsilon

    def calc_w_denominator(rv_vec):
        if rv_vec.ndim != 1:
            input('Error happens because of dimensions')

        m = len(bias_bits)
        n = rv_vec.size
        p = []
        for i in range(m):
            p_temp = 1
            bits = bias_bits[i]
            for j in range(n):
                if j in bits:
                    p_temp = p_temp * stats.norm.pdf(rv_vec[j], mu_biased, sigma)
                else:
                    p_temp = p_temp * stats.norm.pdf(rv_vec[j], mu, sigma)
            p.append(p_temp)
        return np.mean(p)

    def calc_w_nominator(rv_vecs):
        if rv_vecs.ndim != 1:
            input('Error happens because of dimensions')

        p = 1
        for j in range(n):
            p = p * stats.norm.pdf(rv_vecs[j], mu, sigma)
        return p

    now = datetime.now()
    rng = rnd.default_rng(now.microsecond)

    m = len(bias_bits)
    n = x.shape[1]
    noise = np.zeros([m, n], dtype=float)
    weights = np.zeros([m], dtype=float)
    for i in range(m):
        bits_biasd = bias_bits[i]
        # print(bits_biasd)
        noise[i, :] = rng.normal(mu, sigma, x.shape)
        for bit in bits_biasd:
            noise[i, bit] = rng.normal(mu_biased, sigma, 1)
        denom = calc_w_denominator(noise[i,:])
        num = calc_w_nominator(noise[i,:])
        weights[i] = num / denom

    x_tile = np.tile(x, [m, 1])
    y = x_tile + noise
    return y, weights




