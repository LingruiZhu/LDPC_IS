import numpy as np
from operator import itemgetter
from LDPC import decoder, encoder
from utils import binaryproduct
import commpy.channelcoding as cc


def search_boundary(ts, ldpc_paras):
    """
    return the euclidean distance between the ts and error area
    :param ts: trapping set
    :param ldpc_paras: ldpc parameters
    :return: distance from the error area
    """
    H = ldpc_paras['parity_check_matrix']
    n_bits = 44
    messages = np.zeros([1, n_bits])
    c = encoder(messages, ldpc_paras)
    x = pow(-1, c)

    l_min = 1
    l_max = 3.5
    epsilon = (l_max + l_min)/2
    search_steps = 10

    for i in range(search_steps):
        # print(epsilon)
        # input('press enter to continue')
        for bit in ts:
            x[:, bit] = 1 - epsilon
        c_hat = decoder(x, ldpc_paras)
        syndrome = binaryproduct(H, c_hat.T)
        if np.sum(syndrome) == 0:
            # print('no decoding failure')
            epsilon = epsilon + (l_max - l_min) / pow(2, i+1)
        else:
            # print('decoding failure')
            epsilon = epsilon - (l_max - l_min) / pow(2, i+1)

    de_square = len(ts) * pow(epsilon, 2)

    print('epsilot equats to %f' %epsilon)
    print('length of trapping set equals to %d' %len(ts))
    return de_square


def sort_ts(ts_file, ldpc_paras):
    ts_sort_list = []
    ts_dict = np.load(ts_file, allow_pickle='True').item()
    j = 0
    for key in ts_dict.keys():
        print('===================has finished : %d / %d  classes of ts ================='%(j, len(ts_dict.keys())))
        j = j + 1
        for i in range(len(ts_dict[key])):
            print('current ts: %d / %d' %(i, len(ts_dict[key])))
            ts = ts_dict[key][i]
            de_square = search_boundary(ts, ldpc_paras)
            ts_sort_list.append([ts, de_square])
            print('trapping set size: ' + key)
            print('de square is %f' %de_square)

    ts_sort_list = sorted(ts_sort_list, key=itemgetter(1))
    return ts_sort_list


if __name__ == '__main__':
    # ts_file = 'trapping_set_folder/100loops/ts_SNR_1_dB_10loops.npy'
    ts_file = 'trapping_set_folder/ts_SNR_3_dB_2loops.npy'
    ldpc_paras = cc.get_ldpc_code_params('code_matrix/96_3_963.txt', compute_matrix=True)
    ts_list = sort_ts(ts_file, ldpc_paras)
    print(ts_list)

