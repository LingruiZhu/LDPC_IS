import numpy as np
import pandas as pd
import commpy.channelcoding as cc
from LDPC import encoder, decoder
from channel import awgn_IS, awgn_MC
from utils import binaryproduct


def search_ts(code_alist, cycle_nodes_file, snrs, sim_loops):
    """
    search trapping set of LDPC codes with given
    :param code_alist: alist file describing ldpc code structure
    :param cycle_nodes_file: file containing cycle nodes of ldpc codes
    :return: trapping set of ldpc code
    """

    # get parameters of ldpc
    paras = cc.get_ldpc_code_params(code_alist, compute_matrix=True)
    n_bit = 48
    messages = np.zeros([1,n_bit])

    # get cycle nodes
    cycle_nodes = pd.read_csv(cycle_nodes_file, index_col=0)    # data type: DataFrame

    # create the DataFrame to store the information of trapping set
    ts_df = pd.DataFrame(columns={'row', 'snr', 'sim_index', 'ts_size', 'v_nodes', 'c_nodes', 'bias_bits'})
    index = 0

    for snr in snrs:
        ts_dict = {}
        for i in range(cycle_nodes.shape[0]):
            for j in range(sim_loops):
                print('current state: snr: %d dB, row index: %d, loop index: %d ' %(snr, i, j))
                # coping with data type of cycle nodes
                bias_bits = np.array(cycle_nodes.iloc[i, :])
                bias_bits = bias_bits[np.where(~np.isnan(bias_bits))]
                bias_bits = bias_bits.tolist()

                # encoding, biasing and mapping
                c = encoder(messages, paras)
                for node in bias_bits:
                    c[:, int(node)] = 1
                x = pow(-1, c)

                # add biased noise on bias bits
                # y, weights = awgn_IS(x, snr, bias_bits, mode='vs')
                y = awgn_MC(x.astype(float), snr) + 1
                y[x == 1] = 1    # remove noise from normal bits
                # print(y)

                # decoding
                c_hat = decoder(y, paras, max_iter=100)

                H = paras['parity_check_matrix']
                syndrome = binaryproduct(H, c_hat.T)

                # calculating trapping set and degree
                if np.sum(syndrome) != 0:
                    ts_v_bits = np.where(c_hat != 0)[1]
                    ts_v_degree = np.sum(c_hat)

                    ts_c_degree = np.sum(syndrome)
                    ts_size = [ts_v_degree, ts_c_degree]
                    ts_c_bits = np.where(syndrome != 0)[0]

                    info_dict = {'row': i, 'snr': snr, 'sim_index': j,'ts_size': ts_size, 'v_nodes': ts_v_bits,
                                 'c_nodes': ts_c_bits, 'bias_bits': bias_bits}

                    ts_df.loc[index] = info_dict
                    index = index + 1

                    if str(ts_size) in ts_dict.keys():
                        if ts_v_bits.tolist() not in ts_dict[str(ts_size)]:
                            ts_dict[str(ts_size)].append(ts_v_bits.tolist())
                    else:
                        ts_dict[str(ts_size)] = []
                        ts_dict[str(ts_size)].append(ts_v_bits.tolist())
                    print(f'trapping_set: {ts_v_bits.tolist()}')
                    # input('press enter to continue')
            dict_file = 'trapping_set_folder/ts_SNR_' + str(snr) + '_dB_' + str(sim_loops) + 'loops.npy'
            np.save(dict_file, ts_dict)

    save_path = 'TS_info/ts_all_' + str(sim_loops) + 'loops.csv'
    ts_df.to_csv(save_path)

    return ts_df


def ts_spectrum(dict_file):
    ts_dict = np.load(dict_file, allow_pickle='True').item()
    ts_spec_dict = dict()
    for key in ts_dict.keys():
        ts_spec_dict[key] = len(ts_dict[key])
    return ts_spec_dict


if __name__ == '__main__':
    code_para_file = 'code_matrix/96_3_963.txt'
    cycle_file = 'cycles_info/cycles.csv'
    # snrs = np.linspace(0, 10, 11)
    snrs = [5]
    n_loops = 30
    ts = search_ts(code_para_file, cycle_file, snrs, n_loops)

    # dict_file = 'trapping_set_folder/100loops/ts_SNR_3_dB_100loops.npy'
    # dict1 = ts_spectrum(dict_file)
    # print(dict1)
    # print(len(dict1))
    #
    # ts_dict = np.load(dict_file, allow_pickle='True').item()
    # print(ts_dict.keys())
    #
    # print(ts_dict['[4.0, 4.0]'])