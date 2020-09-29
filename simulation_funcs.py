import numpy as np
import commpy.channelcoding as cc
from channel import awgn_IS, awgn_MC
from LDPC import encoder, decoder
from evaluation import fer_MC, fer_IS, ber_MC, ber_IS


def ldpc_simulation(snr, bias_bits, paras, n_trials):
    n_bits = 48
    messages = np.zeros([n_trials, n_bits], dtype=float)

    # encoding, biasing and mapping
    c = encoder(messages, paras)
    # for node in bias_bits:
    #     c[:, int(node)] = 1
    x = pow(-1, c)
    x = x.astype(float)

    y_IS, IS_weights = awgn_IS(x, snr, bias_bits)
    y_MC = awgn_MC(x, snr)

    # decoding
    c_hat_IS = decoder(y_IS, paras, max_iter=100)
    c_hat_MC = decoder(y_MC, paras, max_iter=100)

    # evaluating
    IS_fer, IS_cv = fer_IS(c, c_hat_IS, IS_weights)
    MC_fer, MC_cv = fer_MC(c, c_hat_MC)
    IS_ber = ber_IS(c, c_hat_IS, IS_weights)
    MC_ber = ber_MC(c, c_hat_MC)

    return IS_fer, IS_ber, IS_cv, MC_fer, MC_ber, MC_cv


if __name__ == '__main__':
    bias_bits1 = [30, 36, 51, 88]   # (4,2) trapping set
    bias_bits2 = [44, 53, 58, 92]   # (4,2) trapping set
    # bias_bits2 = [15, 25, 60, 69, 82]

    snr = 7
    ldpc_paras = cc.get_ldpc_code_params('code_matrix/96_3_963.txt', compute_matrix=True)
    n_trials = 300

    IS_fer1, IS_ber1, IS_cv1, MC_fer1, MC_ber1, MC_cv1 = ldpc_simulation(snr, bias_bits1, ldpc_paras, n_trials)
    IS_fer2, IS_ber2, IS_cv2, MC_fer2, MC_ber2, MC_cv2 = ldpc_simulation(snr, bias_bits2, ldpc_paras, n_trials)

    print('result of 1st trapping set')
    print(IS_ber1, MC_ber1, IS_fer1, MC_fer1)
    print('result of 2nd trapping set')
    print(IS_ber2, MC_ber2, IS_fer2, MC_fer2)


