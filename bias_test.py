import numpy as np
import pandas as pd
import commpy.channelcoding as cc
from alist import alist2sparse
from LDPC import encoder, decoder
from utils import binaryproduct
from channel import awgn_MC

ldpc_code_params = cc.ldpc.get_ldpc_code_params('code_matrix/96_3_963.txt', compute_matrix=True)
H = alist2sparse('code_matrix/96_3_963.txt')

n_bits = 48
messages = np.zeros([1, n_bits], dtype=float)

# bias_bits = [1, 2, 3]
# bias_bits = [0, 22, 72, 81, 95]
# bias_bits = [1, 17, 27, 42, 44, 45, 57, 58, 68, 89, 90, 33, 34, 35, 59, 60]


file_path = 'cycles_info/cycles.csv'
cycle_nodes = pd.read_csv(file_path, index_col=0)

#snrs = np.linspace(0, 30, 20)
snrs = [3]
bias = -0.05

for snr in snrs:
    # try to bias each row of cycle nodes
    trapping_set = []
    for i in range(cycle_nodes.shape[0]):
        print('================ current row %d ================' %i)
        bias_bits = np.array(cycle_nodes.iloc[i,:])
        bias_bits = bias_bits[np.where(~np.isnan(bias_bits))].astype(int)
        bias_bits = bias_bits.tolist()

        c = encoder(messages, ldpc_code_params)
        for node in bias_bits:
            c[:, int(node)] = 1
        # print(c)

        x = pow(-1, c)
        # print(x)
        x = x.astype(float)

        y = awgn_MC(x.astype(float), snr) + 1   # mean transition to 0
        # y = bias * np.ones(x.shape)

        y[x == 1] = 1
        print(y)
        y = y.astype(float)
        # print(x)
        # print(y)

        # input('check on the channel output')

        c_hat = decoder(y, ldpc_code_params, max_iter=1000)

        if np.sum(c_hat) != 0:
            syndrome = binaryproduct(H, c_hat.T)
            check_degree_odd = np.sum(syndrome)
            variable_degree = np.sum(c_hat)
            t_size = [int(variable_degree), int(check_degree_odd)]
            error_bits = np.where(c_hat != 0)[1]
            trapping_set.append([t_size, error_bits])

            print('===================== decode failure happels =====================')
            print('codeword c:')
            print(c)
            print('channel vector y:')
            print(y)
            print('decoded:')
            print(c_hat)
            print('bias bits:')
            print(bias_bits)
            print('error bits:')
            print(error_bits)
            print('odd degree check nodes:')
            print(check_degree_odd)
            input('check on ts and bias bits')


    trap_df = pd.DataFrame(trapping_set)
    # save_path = 'TS_info/SNR' + str(snr) + 'dB_' + 'ts_2.csv'
    save_path = 'TS_info/bias_ts_' + str(bias) + '.csv'

    trap_df.to_csv(save_path)




