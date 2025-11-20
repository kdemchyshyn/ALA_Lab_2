import numpy as np

def getEigen(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return 1

    result = list()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    for i in range(len(eigenvalues)):
        value = eigenvalues[i]

        Ax = matrix @ eigenvectors[:, i]
        Vx = value * eigenvectors[:, i]

        diff = Ax - Vx
        if not np.allclose(diff, 0, 1e-10):
            return 1

        result.append((np.real(value), eigenvectors[i]))

    return result