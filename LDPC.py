import commpy.channelcoding as cc
import numpy as np
import utils


def encoder(bits, ldpc_code_params):
    """
    Encode messages with the given parity check matrix
    :param bits:  message to be encoded
    :param H: parity check matrix
    :return:
    codewords (if it works well)
    """
    C = bits.shape[0]    # number of messages to be encoded

    # cc.ldpc.write_ldpc_params(H, 'ldpc_para.txt')
    # ldpc_code_params = []
    # ldpc_code_params = cc.ldpc.get_ldpc_code_params('ldpc_para.txt', compute_matrix=True)
    H = ldpc_code_params['parity_check_matrix'].toarray()
    G = ldpc_code_params['generator_matrix'].toarray()

    m,n = H.shape    # m: number of parity checks  n: number variables nodes
    cword = np.zeros([C, n], dtype=int)

    for c in range(C):
        rhs = bits[c]
        print(rhs.shape)
        cword[c] = cc.ldpc.triang_ldpc_systematic_encode(bits[c], ldpc_code_params)
        lhs = 0

    return cword


def check_codeword(cwords, H):
    C = cwords.shape[0]
    for c in range(C):
        syndrom = sum(H.dot(cwords[c]).reshape(-1, order='F')%2)
        if syndrom == 0:
            flag = True
        else:
            flag = False
        print("valid codeword: ", flag)


def decoder(cwords, ldpc_params, max_iter=30, decoder='SPA'):
    C = cwords.shape[0]
    cwords_de = 2*np.ones(cwords.shape)
    for c in range(C):
        cwords_de[c, :], _ = cc.ldpc_bp_decode(cwords[c, :].reshape(-1, order='f').astype(float), ldpc_params, decoder, max_iter)
    return cwords_de


def get_message(tG, x):
    """Compute the original `n_bits` message from a `n_code` codeword `x`.

    Parameters
    ----------
    tG: array (n_code, n_bits) coding matrix tG.
    x: array (n_code,) decoded codeword of length `n_code`.

    Returns
    -------
    message: array (n_bits,). Original binary message.

    """
    n, k = tG.shape

    rtG, rx = utils.gausselimination(tG, x)

    print('generating matrix')
    print(rtG - tG)

    print('binary message')
    print(rx - x)

    input('check on get message and press enter to continue')

    message = np.zeros(k).astype(int)

    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= utils.binaryproduct(rtG[i, list(range(i+1, k))],
                                          message[list(range(i+1, k))])
    return abs(message)