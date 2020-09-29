import numpy as np
import commpy.channelcoding as cc
import os
import matplotlib.pyplot as plt

from LDPC import encoder, decoder
from alist import alist2sparse
from channel import awgn_IS, awgn_MC
from search_cycles import search_cycles, search_trapping_set
from evaluation import fer_MC, fer_IS, ber_IS, ber_MC

# set parameters of simulation
n_trials = 96    # number of simulation per snr number
n_bits = 48       # length of bit messages
snrs = np.linspace(0, 15, 16)

# read and get ldpc parameters
file_H = 'code_matrix/96_3_963.txt'
H = alist2sparse(file_H)

# cycles, cycles_node = search_cycles(H, 8)
# trap_set = search_trapping_set(cycles, H)

# print(trap_set)
# print(len(trap_set))
# input('check on the trapping set')
#
# print(cycles_node)
# input('here to check nodes in cycles')

cc.ldpc.write_ldpc_params(H, 'ldpc_para.txt')
ldpc_code_params = cc.ldpc.get_ldpc_code_params('ldpc_para.txt', compute_matrix=True)
os.remove('ldpc_para.txt')


# generate message bits
b_temp = np.zeros(48)
b = np.tile(b_temp, (n_trials, 1))
print(b.shape)

# encode messages
c = encoder(b, ldpc_code_params)
x = pow(-1, c)

bers = []
fers = []
cvs = []
cvs_IS = []
fers_IS = []
bers_IS = []
for snr in snrs:
    print('current snr %d' %snr)
    # Monte carlo
    f_err = 0.   # frame error
    y = awgn_MC(x, snr)
    c_hat = decoder(y, ldpc_code_params)

    # for i in range(n_trials):
    #     if np.sum(abs(c_hat[i,:]) - c[i,:]):
    #         f_err = f_err + 1
    # fer = f_err / n_trials
    fer, cv = fer_MC(c_hat, c)
    fers.append(fer)
    cvs.append(cv)

    ber = ber_MC(c, c_hat)
    bers.append(ber)

    # importance sampling
    bias_bits = [0, 11, 21, 64, 2, 29, 83, 29, 37, 64, 81]
    y_IS, weights = awgn_IS(x ,snr, bias_bits, mode='VS')
    c_hat_IS = decoder(y, ldpc_code_params)

    # print(c_hat_IS.shape)
    # print(c_hat.shape)
    # print(weights.shape)
    # input('here to check dimension of weights')

    ber_is = ber_IS(c, c_hat_IS, weights)
    fer_IS_number, cv_IS = fer_IS(c_hat_IS, c, weights)
    fers_IS.append(fer_IS_number)
    cvs_IS.append(cv_IS)
    bers_IS.append(ber_is)

plt.figure()
plt.semilogy(snrs, bers, color='red', label='MC')
plt.semilogy(snrs, bers_IS, color='blue', label='IS')
plt.xlabel('SNR')
plt.ylabel('BER')
plt.title('BER vs SNR')
plt.legend()

plt.figure()
plt.semilogy(snrs, fers, color='red', label='MC')
plt.semilogy(snrs, fers_IS, color='blue', label='IS')
plt.xlabel('SNR')
plt.ylabel('FER')
plt.title('FER vs SNR')
plt.legend()

plt.figure()
plt.semilogy(snrs, cvs, color='red', label='MC')
plt.semilogy(snrs, cvs_IS, color='blue', label='IS')
plt.xlabel('SNR')
plt.ylabel('coefficient of variation')
plt.title('CV vs SNR')
plt.legend()

plt.show()