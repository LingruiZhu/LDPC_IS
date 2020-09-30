import numpy as np
import matplotlib.pyplot as plt
import commpy.channelcoding as cc
from LDPC import encoder, decoder
from utils import binaryproduct
from channel import awgn_MIS, awgn_MC
from evaluation import ber_IS, fer_IS, ber_MC,fer_MC

from base_graph import bg2paras
from Error_boundary import sort_ts


class SimResult:
    def __init__(self, name, snrs):
        self.name = name
        self.snrs = snrs
        self.fers_mean = []
        self.fers_std = []
        self.bers_mean = []
        self.bers_std = []
        self.fers_cv = []
        self.bers_cv = []

    def calc_cv(self):
        self.fers_cv = [a/b for a,b in zip(self.fers_std, self.fers_mean)]
        self.bers_cv = [a/b for a,b in zip(self.bers_std, self.bers_mean)]

    def plot(self, result1):
        plt.figure()
        plt.title('BER vs SNR')
        plt.semilogy(self.snrs, self.bers_mean, label=self.name, color='blue', marker='x')
        plt.semilogy(result1.snrs, result1.bers_mean, label=result1.name, color='red', marker='o')
        plt.xlabel('SNR')
        plt.ylabel('BER')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title('FER vs SNR')
        plt.semilogy(self.snrs, self.fers_mean, label=self.name, color='blue', marker='x')
        plt.semilogy(result1.snrs, result1.fers_mean, label=result1.name, color='red', marker='o')
        plt.xlabel('SNR')
        plt.ylabel('FER')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title('CV of BER vs SNR')
        plt.semilogy(self.snrs, self.bers_cv, label=self.name, color='blue', marker='x')
        plt.semilogy(result1.snrs, result1.bers_cv, label=result1.name, color='red', marker='o')
        plt.xlabel('SNR')
        plt.ylabel('Coefficient of Variation for BER')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title('CV of FER vs SNR')
        plt.semilogy(self.snrs, self.fers_cv, label=self.name, color='blue', marker='x')
        plt.semilogy(result1.snrs, result1.fers_cv, label=result1.name, color='red', marker='o')
        plt.xlabel('SNR')
        plt.ylabel('Coefficient of Variation for FER')
        plt.legend()
        plt.grid()
        plt.show()


def trapping_set_search(alist_file, save_ts=False):
    ldpc_paras = cc.get_ldpc_code_params(alist_file, compute_matrix=True)
    n_v = ldpc_paras['n_vnodes']
    n_c = ldpc_paras['n_cnodes']
    n_bits_message = n_v - n_c
    messages = np.zeros([1, n_bits_message])

    # bias parameters for finding trapping set
    epsilon1 = 3.2
    gamma = 0.75
    epsilon2 = 1 - gamma

    # parameters of LDPC codes

    v_degree_max = ldpc_paras['max_vnode_deg']
    c_degree_max = ldpc_paras['max_cnode_deg']
    v_nodes_adj = ldpc_paras['vnode_adj_list'].reshape(n_v, v_degree_max)
    c_nodes_adj = ldpc_paras['cnode_adj_list'].reshape(n_c, c_degree_max)
    H = ldpc_paras['parity_check_matrix']

    trapping_set_list = []

    ts_dict = {}

    for i in range(n_v):
        # encoding and mapping
        c = encoder(messages, ldpc_paras)
        x = pow(-1, c)
        snr = 3
        snr_linear = pow(10, snr/10)

        # bias according to the rules in 'A general method for finding low error rates of LDPC codes'
        bias_candidate_list = []
        for j in range(v_degree_max):
            c_adj = v_nodes_adj[i, j]
            tier1_v_nodes = np.delete(c_nodes_adj[c_adj], np.where(c_nodes_adj[c_adj] == i))
            bias_candidate_list.append(tier1_v_nodes)

        print(bias_candidate_list)
        # input('press enter to continue. hava a check on bias bits')

        for bias_bit1 in list(bias_candidate_list[0]):
            for bias_bit2 in list(bias_candidate_list[1]):
                for bias_bit3 in list(bias_candidate_list[2]):
                    bias_bits = [i, bias_bit1, bias_bit2, bias_bit3]

                    # bias_bits = [100, 67, 78, 113]

                    # bias the input of decoder
                    x = gamma * np.ones(x.shape)
                    for bit in bias_bits:
                        x[:, bit] = 1 - epsilon1
                    llr_x = x * 4 * snr_linear
                    print(llr_x)
                    c_hat = decoder(llr_x, ldpc_paras, decoder='MSA', max_iter=50)

                    print('now check sum is %f' %np.sum(c_hat))
                    # input('have a check here on parity checks')
                    if np.sum(c_hat) != 0:
                        a = np.sum(c_hat)
                        b = np.sum(binaryproduct(H, c_hat.T))
                        ts_v_bits = np.where(c_hat != 0)[1]
                        ts_size = [a, b]
                        trapping_set_list.append([ts_size, ts_v_bits])
                        print(ts_size)
                        print(ts_v_bits)
                        print(bias_bits)
                        # input('press enter to continue')

                        if tuple(ts_size) in ts_dict.keys():
                            if ts_v_bits.tolist() not in ts_dict[tuple(ts_size)]:
                                ts_dict[tuple(ts_size)].append(ts_v_bits.tolist())
                        else:
                            ts_dict[tuple(ts_size)] = []
                            ts_dict[tuple(ts_size)].append(ts_v_bits.tolist())

    if save_ts:
        save_path = 'trapping_set_folder/tuple_general_SNR_' + str(snr) + '_' + str(n_v) + '_' + str(n_v - n_c) + '_' + str(epsilon1) + '_ts_dict.npy'
        np.save(save_path, ts_dict)

    return ts_dict


def general_IS(bias_bits, epsilon, ldpc_paras, snrs):
    n_trials = 200
    n_bits = ldpc_paras['n_vnodes'] - ldpc_paras['n_cnodes']
    message =  np.zeros([1, n_bits])
    c = encoder(message, ldpc_paras)
    x = pow(-1, c)

    fers_mean_snrs = []
    fers_std_snrs = []
    bers_mean_snrs = []
    bers_std_snrs = []
    for snr in snrs:
        snr_linear = pow(10, snr/10)
        fers_per_snr = []
        bers_per_snr = []
        for i in range(n_trials):
            y, weights = awgn_MIS(x, snr, bias_bits, epsilon)

            print('check weights')
            print(weights)

            c_tile = np.tile(c, [len(bias_bits), 1])

            for i in range(len(bias_bits)):
                y_temp = y[i,:]
                weight = [weights[i]]
                llr_input_temp = 4*y_temp*snr_linear
                llr_input_temp = llr_input_temp.reshape((1, y.shape[1]))
                c_hat_temp = decoder(llr_input_temp, ldpc_paras)
                fer = fer_IS(c, c_hat_temp, weight)
                ber = ber_IS(c, c_hat_temp, weight)
                bers_per_snr.append(ber)
                fers_per_snr.append(fer)


            # llr_input = 4*y*snr_linear
            # c_hat = decoder(llr_input, ldpc_paras)
            # ber_is_value = ber_IS(c_tile, c_hat, weights)
            # fer_is_value, _ = fer_IS(c_tile, c_hat, weights)
            # bers_per_snr.append(ber_is_value)
            # fers_per_snr.append(fer_is_value)

        fer_mean = np.mean(fers_per_snr)
        fer_std = np.std(fers_per_snr)
        ber_mean = np.mean(bers_per_snr)
        ber_std = np.std(bers_per_snr)

        fers_mean_snrs.append(fer_mean)
        fers_std_snrs.append(fer_std)
        bers_mean_snrs.append(ber_mean)
        bers_std_snrs.append(ber_std)

    result = SimResult('IS', snrs)
    result.fers_mean = fers_mean_snrs
    result.fers_std = fers_std_snrs
    result.bers_mean = bers_mean_snrs
    result.bers_std = bers_std_snrs
    return result


def monte_carlo(ldpc_paras ,snrs):
    n_trials = 800
    n_bits = ldpc_paras['n_vnodes'] - ldpc_paras['n_cnodes']
    message =  np.zeros([1, n_bits])
    c = encoder(message, ldpc_paras)
    x = pow(-1, c)

    fers_mean_snrs = []
    fers_std_snrs = []
    bers_mean_snrs = []
    bers_std_snrs = []

    for snr in snrs:
        snr_linear = pow(10, snr/10)

        fers_per_snr = []
        bers_per_snr = []

        for i in range(n_trials):
            y = awgn_MC(x, snr)
            llr_input = 4*snr_linear*y
            c_hat = decoder(llr_input, ldpc_paras)
            ber_is_value = ber_MC(c, c_hat)
            fer_is_value, _ = fer_MC(c, c_hat)

            bers_per_snr.append(ber_is_value)
            fers_per_snr.append(fer_is_value)

        fer_mean = np.mean(fers_per_snr)
        fer_std = np.std(fers_per_snr)
        ber_mean = np.mean(bers_per_snr)
        ber_std = np.std(bers_per_snr)

        fers_mean_snrs.append(fer_mean)
        fers_std_snrs.append(fer_std)
        bers_mean_snrs.append(ber_mean)
        bers_std_snrs.append(ber_std)

    result = SimResult('MC', snrs)
    result.fers_mean = fers_mean_snrs
    result.fers_std = fers_std_snrs
    result.bers_mean = bers_mean_snrs
    result.bers_std = bers_std_snrs

    return result


def ts_statics(ts_file, paras):
    ts_set_dict = np.load(ts_file, allow_pickle='True').item()
    for key in ts_set_dict.keys():

        print(key + ': %d' %len(ts_set_dict[key]))

    ts_sort_list = sort_ts(ts_file, paras)
    print(ts_sort_list)



if __name__ == '__main__':
    alist_file = 'code_matrix/bg_mat.txt'
    ts = trapping_set_search(alist_file, save_ts=True)

    # ts_set_file = 'trapping_set_folder/general_SNR_3_136_44_3.5_ts_dict.npy'
    # params = cc.get_ldpc_code_params(alist_file, compute_matrix=True)
    # ts_statics(ts_set_file, params)
    # input('press enter to continue, now check domiant trapping sets')
    #
    # # test importance sampling part
    # bias_bits_list = []
    # bias_bits_list.append([1, 17, 42, 54])
    # bias_bits_list.append([1, 44, 57, 89, 92])
    # bias_bits_list.append([5, 30, 60, 66])
    # bias_bits_list.append([12, 61, 79])
    #
    # #epsilon = 1.9982
    # epsilon = 1.5
    # snrs = np.linspace(0, 5, 6)
    #
    # result_is = general_IS(bias_bits_list, epsilon, params, snrs)
    # result_mc = monte_carlo(params, snrs)
    #
    # # calculate coefficient of variation
    # result_is.calc_cv()
    # result_mc.calc_cv()
    #
    # print('FERS mean:')
    # print(result_is.fers_mean)
    # print(result_mc.fers_mean)
    #
    # print('BERs mean:')
    # print(result_is.bers_mean)
    # print(result_mc.bers_mean)
    #
    # print('FERs std:')
    # print(result_is.fers_std)
    # print(result_mc.fers_std)
    #
    # print('BERs std:')
    # print(result_is.bers_std)
    # print(result_mc.bers_std)
    #
    # result_is.plot(result_mc)



