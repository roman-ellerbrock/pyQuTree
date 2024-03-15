import numpy as np

def givens_rotation(B, c, s, i, j):
    """
    Perform a Givens-Rotation: R*A*R^H with R= {{c, s}, {-s*, c}}
    """
    copy0 = (c * B[:, :, i] + s * B[:, :, j]).copy()
    copy1 = (c * B[:, :, j] - np.conj(s) * B[:, :, i]).copy()
    B[:, :, i] = copy0
    B[:, :, j] = copy1
    copy0 = B[:, i, :].copy()
    copy1 = B[:, j, :].copy()
    B[:, i, :] = c * copy0 + np.conj(s) * copy1
    B[:, j, :] = c * copy1 - s * copy0

def build_G_matrix(A, i, j):
    """
    Build the G-Matrix
    """
    M = A.shape[0]
    G = np.zeros((3, 3), dtype=np.complex128)
    h = np.zeros((M, 3), dtype=np.complex128)

    # Build h-Vector for this Matrix
    h[:, 0] = A[:, i, i] - A[:, j, j]
    h[:, 1] = A[:, i, j] + A[:, j, i]
    h[:, 2] = 1j * (A[:, j, i] - A[:, i, j])

    # Add to G-Matrix
#    for m in range(M):
#        for n in range(3):
#            for o in range(3):
#                G[n, o] += np.conj(h[m, n]) * h[m, o]
    G = np.einsum('mi,mj->ij', np.conj(h), h) 
    return G

def calculate_angles(A, i, j):
    """
    Calculate the angles c and s for the Givens-Rotation
    """
    # Build the G-Matrix
    G = build_G_matrix(A, i, j)

    # Diagonalize it (phase convention is important here (x_>0)!)
    _, trafo = np.linalg.eigh(G)

    # Calculate the angles from the
    x = trafo[0, 2]
    y = trafo[1, 2]
    z = trafo[2, 2]

    r = np.sqrt(x * x + y * y + z * z)
    t = np.sqrt(2. * r * (x + r))
    if np.abs(t) < 1e-14:
        return 1., 0.

    c = (x + r) / t
    s = (y - 1j * z) / t
    if np.isrealobj(A):
        c = np.abs(c)
        s = np.abs(s)
    norm = c * c + s * s
    c /= np.sqrt(norm)
    s /= np.sqrt(norm)
    return c, s

def givens_trafo_rotation(U, c, s, i, j):
    """
    Perform a Givens-Rotation on the Transformation-Matrix
    """
    Ui = U[:, i].copy()
    Uj = U[:, j].copy()
    copy0 = (c * Ui + s * Uj)
    copy1 = (c * Uj - np.conj(s) * Ui)
    U[:, i] = copy0.copy()
    U[:, j] = copy1.copy()
    return U

def jacobi_rotations(A, U):
    """
    Perform Jacobi-Rotations to simultaneously diagonalize a set of matrices
    """
    # Angles for Givens-Rotation
    c, s = 0, 0

    # Swipe over the matrix-dimensions and perform jacobi-rotations
    for i in range(A.shape[1]):
        for j in range(i + 1, A.shape[2]):
            # Calculate Angles c and s for the elements i and j
            c, s = calculate_angles(A, i, j)

            # Perform the Givens-Rotation with angles c and s
            givens_rotation(A, c, s, i, j)

            # Rotate the Transformation-Matrix
            givens_trafo_rotation(U, c, s, i, j)

def off_diagonal_measure(A):
    A = A.copy()
    result = 0.
    for i in range(A.shape[0]):
        np.fill_diagonal(A[i], 0.)
        result += np.linalg.norm(A[i])
    return result

def simultaneous_diagonalization(A, n_iter = 100, eps = 1e-12, verbose = False):
    """
    A: np.ndarray with shape (k, m, m)
    k: number of matrices to diagonalize simultaneously
    """
    if len(A.shape) != 3:
        raise ValueError('A must be a 3D tensor.')

    U = np.eye(A.shape[1])
    for it in range(n_iter):
        jacobi_rotations(A, U)
        delta = off_diagonal_measure(A)
        if verbose:
            print('Iteration: {} Delta: {}'.format(it, delta))
        if delta < eps:
            break

    return A, U