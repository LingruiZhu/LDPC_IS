import numpy as np
import scipy
import scipy.sparse


def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2


def gausselimination(A, b):
    """Solve linear system in Z/2Z via Gauss Gauss elimination."""
    if type(A) == scipy.sparse.csr_matrix:
        A = A.toarray().copy()
    else:
        A = A.copy()
    b = b.copy()
    n, k = A.shape

    for j in range(min(k, n)):
        listedepivots = [i for i in range(j, n) if A[i, j]]
        if len(listedepivots):
            pivot = np.min(listedepivots)
        else:
            continue
        if pivot != j:
            aux = (A[j, :]).copy()
            A[j, :] = A[pivot, :]
            A[pivot, :] = aux

            aux = b[j].copy()
            b[j] = b[pivot]
            b[pivot] = aux

        for i in range(j+1, n):
            if A[i, j]:
                A[i, :] = abs(A[i, :]-A[j, :])
                b[i] = abs(b[i]-b[j])

    return A, b


def get_matrix(N, K):
    file_H = 'code_matrix/LDPC_chk_mat_' + str(N) + '_' + str(N-K) + '.txt'
    file_G = 'code_matrix/LDPC_gen_mat_' + str(N) + '_' + str(N-K) + '.txt'

    # file_H = '/home/jianping/PycharmProjects/pyldpc-master/LDPC_matrix/LDPC_chk_mat_155_62.txt'
    # file_G = '/home/jianping/PycharmProjects/pyldpc-master/LDPC_matrix/LDPC_gen_mat_155_62.txt'

    H_matrix_row_col = np.loadtxt(file_H, dtype=np.int32)
    G_matrix_row_col = np.loadtxt(file_G, dtype=np.int32)

    H_mat = np.zeros([K, N], dtype=np.int32)
    G_mat = np.zeros([N-K, N], dtype=np.int32)

    H_mat[H_matrix_row_col[:,0], H_matrix_row_col[:,1]] = 1
    G_mat[G_matrix_row_col[:,0], G_matrix_row_col[:,1]] = 1

    return H_mat, G_mat.T


def _bitsandnodes(H):
    """Return bits and nodes of a parity-check matrix H."""
    if type(H) != scipy.sparse.csr_matrix:
        bits_indices, bits = np.where(H)
        nodes_indices, nodes = np.where(H.T)
    else:
        bits_indices, bits = scipy.sparse.find(H)[:2]
        nodes_indices, nodes = scipy.sparse.find(H.T)[:2]
    bits_histogram = np.bincount(bits_indices)
    nodes_histogram = np.bincount(nodes_indices)

    return bits_histogram, bits, nodes_histogram, nodes