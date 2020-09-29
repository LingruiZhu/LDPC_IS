import numpy as np


def alistToNumpy(lines):
    """Converts a parity-check matrix in AList format to a 0/1 numpy array. The argument is a
    list-of-lists corresponding to the lines of the AList format, already parsed to integers
    if read from a text file.
    The AList format is introduced on http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html.
    This method supports a "reduced" AList format where lines 3 and 4 (containing column and row
    weights, respectively) and the row-based information (last part of the Alist file) are omitted.
    Example:
        alistToNumpy([[3,2], [2, 2], [1,1,2], [2,2], [1], [2], [1,2], [1,2,3,4]])
        array([[1, 0, 1],
               [0, 1, 1]])
    """
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=np.int)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix


def alist2sparse(file_H):
    with open(file_H) as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            new_lines.append(list(map(int, line.split())))
    H_matrix = alistToNumpy(new_lines)
    return H_matrix