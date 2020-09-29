import numpy as np
import matplotlib.pyplot as plt
import commpy.channelcoding as cc

from search_ts import ts_spectrum
from LDPC import encoder, decoder
from channel import awgn_MC, awgn_IS
from evaluation import ber_IS, ber_MC


def single_ts_sim(ts, snrs, n_trials , ldpc_params):
    n_bit = 48
    messages = np.zeros([n_trials, n_bit])

    bers_IS = []
    bers_MC = []
    for snr in snrs:
        c = encoder(messages, ldpc_params)
        x = pow(-1, c)

        y_IS, weights = awgn_IS(x, snr, ts, mode='VS')
        y_MC = awgn_MC(x, snr)

        c_hat_IS = decoder(y_IS, ldpc_params)
        c_hat_MC = decoder(y_MC, ldpc_params)

        ber_is = ber_IS(c_hat_IS, c, weights)
        ber_mc = ber_MC(c_hat_MC, c)

        bers_IS.append(ber_is)
        bers_MC.append(ber_mc)

    return bers_IS, bers_MC


def IS_ts_sim(snr, n_trails_per_ts, ldpc_paras):
    ts_file = 'trapping_set_folder/100loops/ts_SNR_' + str(int(snr)) + '_dB_100loops.npy'
    ts_dict = np.load(ts_file, allow_pickle='True').item()
    ts_spct_dict = ts_spectrum(ts_file)
    bers_is_classes = []
    weights = []
    for key in ts_dict.keys():
        ts = ts_dict[key][0]
        ber_is_element = ts_single_IS(snr, ts, n_trails_per_ts, ldpc_paras)
        bers_is_classes.append(ber_is_element)
        weights.append(ts_spct_dict[key])
    bers = np.array(bers_is_classes)
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    ber_is_per_snr = np.sum(bers * weights)
    num_classes = len(bers_is_classes)
    return ber_is_per_snr, num_classes



def plot_different_ts():
    ts0 = [0, 22, 81, 95]
    ts1 = [0, 72, 81, 95]
    ts2 = [2, 26, 29, 40]
    ts3 = [5, 46, 72, 88]
    snrs = np.linspace(5,10,6)

    ldpc_paras = cc.get_ldpc_code_params('code_matrix/96_3_963.txt', compute_matrix='True')
    bers_IS0, bers_MC0 = single_ts_sim(ts0, snrs, 100, ldpc_paras)
    bers_IS1, bers_MC1 = single_ts_sim(ts1, snrs, 100, ldpc_paras)
    bers_IS2, bers_MC2 = single_ts_sim(ts2, snrs, 100, ldpc_paras)
    bers_IS3, bers_MC3 = single_ts_sim(ts3, snrs, 100, ldpc_paras)

    # plot
    plt.figure()
    plt.semilogy(snrs, bers_MC0, color='red', marker='o', label='ts 0')
    plt.semilogy(snrs, bers_MC1, color='blue', marker='*', label='ts 1')
    plt.semilogy(snrs, bers_MC2, color='green', marker='d', label='ts 2')
    plt.semilogy(snrs, bers_MC3, color='orange', marker='p', label='ts 3')
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title('Monte Carlo BER vs SNR')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogy(snrs, bers_IS0, color='red', marker='o', ls='--', label='ts 0')
    plt.semilogy(snrs, bers_IS1, color='blue', marker='*', ls='--', label='ts 1')
    plt.semilogy(snrs, bers_IS2, color='green', marker='d', ls='--', label='ts 2')
    plt.semilogy(snrs, bers_IS3, color='orange', marker='p', ls='--', label='ts 3')
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title('Importance sampling BER vs SNR')
    plt.grid()
    # plt.legend()
    plt.show()


def ts_single_IS(snr_value, ts, n_trials, ldpc_paras):
    n_bit = 48
    messages = np.zeros([n_trials, n_bit])
    c = encoder(messages, ldpc_paras)
    # for bit in ts:
    #     c[:, bit] = np.ones([n_trials, ])
    x = pow(-1, c)
    y, weights = awgn_IS(x, snr_value, ts, mode='MT')
    # y[x==1] = 1
    # print(y)
    # input('have a check on decoder input y')
    c_hat = decoder(y, ldpc_paras)
    ber_is = ber_IS(c_hat, c, weights)
    return ber_is


if __name__ == '__main__':
    snrs = np.linspace(0, 10, 11)
    ldpc_paras = cc.get_ldpc_code_params('code_matrix/96_3_963.txt', compute_matrix='Ture')
    bers_IS = []
    number_classes = []
    n_trials = 20
    for snr in snrs:
        print('========= current simulated snr: %d ================='%snr)
        ber_is, n_class = IS_ts_sim(snr, n_trials, ldpc_paras)
        print('number of ts classes: %d' %n_class)
        bers_IS.append(ber_is)
        number_classes.append(n_class)

    print(bers_IS)

    plt.figure()
    plt.semilogy(snrs, bers_IS, marker='d', color='blue', label='IS')
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title('BER vs SNR')
    plt.legend()
    plt.grid()
    plt.show()

