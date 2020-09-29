import numpy as np
import commpy.channelcoding as cc


def bg2paras(bg_file, Zc, paras_file):
    """
    convert base graph file to alist parameters
    :param paras_file: alist file path
    :param bg_file: base graph file
    :param Zc: expansion factor
    :return: check matrix and alist parameters
    """
    B = np.loadtxt(bg_file, dtype=int)
    [m, n] = B.shape
    H = np.zeros([Zc * m, Zc * n], dtype=int)
    for i in range(m):
        for j in range(n):
            identity = np.identity(Zc)
            if B[i, j] == -1:
                identity = np.zeros([Zc, Zc])
            else:
                identity = np.roll(identity, B[i, j], axis=1)
            # print(identity)

            H[i * Zc:i * Zc + Zc, j * Zc:j * Zc + Zc] = identity

    cc.ldpc.write_ldpc_params(H, paras_file)
    paras = cc.get_ldpc_code_params(paras_file, compute_matrix=True)
    return H, paras


if __name__ == '__main__':
    # test
    bg_file = 'base_matrices/NR_1_0_2.txt'
    Zc = 2
    alist_path = 'code_matrix/test.txt'
    H, ldpc_paras = bg2paras(bg_file, Zc, alist_path)
    print(H.shape)