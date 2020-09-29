import numpy as np
import commpy.channelcoding as cc
import os
from numpy.random import choice
from alist import alist2sparse
from search_cycles import search_cycles
from LDPC import encoder, decoder
from utils import binaryproduct

from decoder import decode


file_H = 'code_matrix/96_3_963.txt'
H = alist2sparse(file_H)
np.save('H_matrix.txt',H)


print(H)
input('check on the H')

cc.ldpc.write_ldpc_params(H, 'ldpc_para.txt')
ldpc_code_params = cc.ldpc.get_ldpc_code_params('ldpc_para.txt', compute_matrix=True)
os.remove('ldpc_para.txt')
# input('here to check')
cycles, cycle_nodes = search_cycles(H, L=6)
# print(cycles)
print(cycle_nodes)

# cycle_nodes = [[10], [20]]

input('press enter to continue and check on cycle nodes')

row_number = len(cycle_nodes)
for i in range(row_number):
    nodes = cycle_nodes[i]
    nodes_length = len(nodes)

    # single binary message simulation
    n_bits = 48
    messages = np.zeros([1, n_bits], dtype=float)
    c = encoder(messages, ldpc_code_params)
    print('codeword after encoder')
    print(c)

    # add bias value before modulation
    # for j in range(nodes_length):
    #     c[:, nodes[j]] = 1
    #
    # print('codeword after bias')
    # print(c)

    x = pow(-1, c)
    x = x.astype(np.float)
    # add bias value after modulation
    for j in range(nodes_length):
        x[:, nodes[j]] = - 0.2

    x = x + np.random.uniform(-0.01, +0.01, 96)
    print(x)

    # decoder
    c_hat = decoder(x, ldpc_code_params, max_iter=1000)
    # c_hat = decode(H, x)
    input('have a check on the c_hat')

    # check if error happens
    if np.sum(c_hat) != 0:
        print(x)
        print(c_hat)
        syndrome = binaryproduct(H, c_hat.T)
        trapping_nv = nodes_length
        trapping_nc = np.sum(syndrome)
        trapping_set = np.nonzero(c_hat)
        print([trapping_nv, trapping_nc])
        print(trapping_set)
        input('have a check on trapping set')


