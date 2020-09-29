import numpy as np
import commpy.channelcoding as cc
import numpy.random as rnd
from datetime import datetime
from commpy.utilities import signal_power
from LDPC import encoder, decoder


def ts_test(paras, bias_set, snr):
    n_bits = 48
    messages = np.zeros([1, n_bits])
    c = encoder(messages, paras)
    x = pow(-1, c)

    # channel and noise
    sig_power = signal_power(x)
    now = datetime.now()
    rng = rnd.default_rng(now.second)
    n_power = sig_power * (10 ** (-snr / 20))
    mu = 0
    normal_noise = rng.normal(mu, n_power, x.shape)

    y1 = x + normal_noise
    y2 = x.astype(float)

    mu_biased = -1
    bias_noise = rng.normal(mu_biased, n_power, len(bias_set))
    for i in range(len(bias_set)):
        node = bias_set[i]
        y1[:, node] = x[:, node] + bias_noise[i]
        y2[:, node] = x[:, node] + bias_noise[i]

    print(bias_noise)
    print(y1)
    print(y2)

    # decoder
    c_hat1 = decoder(y1, paras, max_iter=100)
    c_hat2 = decoder(y2, paras, max_iter=100)

    error_bits1 = np.where(c_hat1 != 0)[1]
    error_bits2 = np.where(c_hat2 != 0)[1]

    return error_bits1, error_bits2


if __name__ == '__main__':
    paras = cc.get_ldpc_code_params('code_matrix/96_3_963.txt', compute_matrix=True)
    ts_set = [8, 36, 51, 76, 84]
    snr = 3
    e_bits1, e_bits2 = ts_test(paras, ts_set, snr)
    print(e_bits1)
    print(e_bits2)
