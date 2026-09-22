
import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray


def cubic_eigen_solver(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> tuple:
    """
    This function solves the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3)p=0 and computes the vertical atmospheric
    (VA) mode shapes p and the modal eigenvalues λ
    Before being solved, the cubic eigenvalue is first turned into a generalized eigenvalue problem Xp̃ = λYp̃.

    Parameters
    ----------
    a: NDArray
    Matrix A in the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3).
    b: NDArray
    Matrix B in the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3).
    c: NDArray
    Matrix C in the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3).
    d: NDArray
    Matrix D in the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3).

    Returns
    ----------
    tuple[NDArray, NDArray]
    """

    vals = []
    vecs = []

    n = len(a)
    if np.linalg.norm(d, 1) != 0:  # Checking if the eigenvalue problem is of third order

        # Creating Matrix X (X = [[A, 0, 0], [0, C^T, 0], [0, 0, D^T]])
        aux = np.zeros((n, n))
        x = np.concatenate((a, aux, aux), axis=1)
        x1 = np.concatenate((aux, np.transpose(c), aux), axis=1)
        x = np.concatenate((x, x1), axis=0)
        x1 = np.concatenate((aux, aux, np.transpose(d)), axis=1)
        x = np.concatenate((x, x1), axis=0)

        # Creating Matrix Y (Y = [[B, C, D], [C^T, 0, 0], [0, D^T, 0]])
        y = np.concatenate((b, c, d), axis=1)
        y1 = np.concatenate((np.transpose(c), aux, aux), axis=1)
        y = np.concatenate((y, y1), axis=0)
        y1 = np.concatenate((aux, np.transpose(d), aux), axis=1)
        y = np.concatenate((y, y1), axis=0)

        # Computing the eigenvectors and eingenvalues
        iy = inv(y)
        u = np.matmul(iy, x)

        vals, vecs = np.linalg.eig(u)

        # deleting  γp and (γ^2)p in [[p], [γp], [(γ^2)p]]
        vecs = np.delete(vecs, range(3 * n - 1, n - 1, -1), 0)

        # Normalizing vectors (Unitary L2 Norm)
        for i in range(0, len(vecs[0, :])):
            norm = np.sum(1/(len(vecs[:, i])) * abs(vecs[:, i]))
            vecs[:, i] = (1 / norm) * vecs[:, i]

    if np.linalg.norm(d, 1) == 0 and np.linalg.norm(c, 1) != 0:  # Checking if the eigenvalue problem is of second order

        # Creating Matrix X (X = [[A, 0], [0, C^T]])
        aux = np.zeros((n, n))
        x = np.concatenate((a, aux), axis=1)
        x1 = np.concatenate((aux, np.transpose(c)), axis=1)
        x = np.concatenate((x, x1), axis=0)

        # Creating Matrix Y (Y = [[B, C], [C^T, 0]])
        y = np.concatenate((b, c), axis=1)
        y1 = np.concatenate((np.transpose(c), aux), axis=1)
        y = np.concatenate((y, y1), axis=0)

        # Computing the eigenvectors and eingenvalues
        iy = inv(y)
        u = np.matmul(iy, x)

        vals, vecs = np.linalg.eig(u)

        # deleting  γp and (γ^2)p in [[p], [γp], [(γ^2)p]]
        vecs = np.delete(vecs, range(2 * n - 1, n - 1, -1), 0)

        # Normalizing vectors (Unitary L2 Norm)
        for i in range(0, len(vecs[0, :])):
            norm = np.sum(1 / (len(vecs[:, i])) * abs(vecs[:, i]))
            vecs[:, i] = (1 / norm) * vecs[:, i]

    if np.linalg.norm(d, 1) == 0 and np.linalg.norm(c, 1) == 0:  # Checking if the eigenvalue problem is of first order

        # Creating Matrix X (X = [A])
        x = a

        # Creating Matrix Y (Y = [B])
        y = b

        iy = inv(y)
        u = np.matmul(iy, x)

        vals, vecs = np.linalg.eig(u)

        # Normalizing vectors (Unitary L2 Norm)
        for i in range(0, len(vecs[0, :])):
            norm = np.sum(1 / (len(vecs[:, i])) * abs(vecs[:, i]))
            vecs[:, i] = (1 / norm) * vecs[:, i]

    return vals, vecs
