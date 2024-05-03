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

def build_G_matrix_real(A, i, j):
    """
    Build the G-Matrix
    """
    M = A.shape[0]
    G = np.zeros((2, 2), dtype=np.float64)
    h = np.zeros((M, 2), dtype=np.complex128)

    # Build h-Vector for this Matrix
    h[:, 0] = A[:, i, i] - A[:, j, j]
    h[:, 1] = A[:, i, j] + A[:, j, i]

    G = np.real(np.einsum('mi,mj->ij', np.conj(h), h))
    return G

def calculate_angles_real(A, i, j):
    """
    Calculate the angles c and s for the Givens-Rotation
    """
    # Build the G-Matrix
    G = build_G_matrix_real(A, i, j)

    # Diagonalize it (phase convention is important here (x_>0)!)
    _, trafo = np.linalg.eigh(G)

    # Calculate the angles from the
    x = trafo[0, 1]
    y = trafo[1, 1]

    r = np.sqrt(x * x + y * y)
    t = np.sqrt(2 * r * (x + r))
    if r == 0 or t == 0:
        return 1., 0.

    c = np.sqrt((x + r) / (2*r))
    s = y  / t

    norm = c * c + s * s
    c /= np.sqrt(norm)
    s /= np.sqrt(norm)
    return c, s

def build_G_matrix(A, i, j):
    """
    Build the G-Matrix
    """
    M = A.shape[0]
    G = np.zeros((3, 3), dtype=np.float64)
    h = np.zeros((M, 3), dtype=np.complex128)

    # Build h-Vector for this Matrix
    h[:, 0] = A[:, i, i] - A[:, j, j]
    h[:, 1] = A[:, i, j] + A[:, j, i]
    h[:, 2] = 1j * (A[:, j, i] - A[:, i, j])

    G = np.real(np.einsum('mi,mj->ij', np.conj(h), h))
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
    t = np.sqrt(2 * r * (x + r))
    if np.abs(r) < 1e-16:
        C = np.array([A[0, i, i], A[0, i, j], A[0, j, i], A[0, j, j]]).reshape((2, 2))
        h = np.zeros((A.shape[0], 3), dtype=np.complex128)
        h[:, 0] = A[:, i, i] - A[:, j, j]
        h[:, 1] = A[:, i, j] + A[:, j, i]
        h[:, 2] = 1j * (A[:, j, i] - A[:, i, j])
        print(i, j, A.dtype)
        print('A[] = ', C)
        print('h = ', h)
        print(G)
        print(x, y, r, t)
        return 1., 0.

    #c = (x + r) / t
    if r == 0 or t == 0:
        return 1., 0.
    c = np.sqrt((x + r) / (2*r))
    s = (y - 1j * z) / t
    if np.isrealobj(A):
        s = y / t
    else:
        s = (y - 1j * z) / t
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

    acc = 0. # accumulated off-angles
    # Swipe over the matrix-dimensions and perform jacobi-rotations
    for i in range(A.shape[1]):
        for j in range(i + 1, A.shape[2]):
            # Calculate Angles c and s for the elements i and j
            c, s = 1., 0.
            if np.isrealobj(A):
                print('real')
                c, s = calculate_angles_real(A, i, j)
            else:
                print('complex')
                c, s = calculate_angles(A, i, j)
            acc += np.abs(s) # add rotations of off-diagonal elements

            # Perform the Givens-Rotation with angles c and s
            givens_rotation(A, c, s, i, j)

            # Rotate the Transformation-Matrix
            givens_trafo_rotation(U, c, s, i, j)
    return acc

def off_diagonal_measure(A):
    A = A.copy()
    result = 0.
    for i in range(A.shape[0]):
        np.fill_diagonal(A[i], 0.)
        result += np.linalg.norm(A[i])
    return result

def simultaneous_diagonalization(A, n_iter = 100, eps = 1e-12, verbose = False, copy = True):
    """
    A: np.ndarray with shape (k, m, m)
    k: number of matrices to diagonalize simultaneously
    eps: convergence criterion
    n_iter: maximum number of iterations
    verbose: print progress of the algorithm
    copy: if True, the input array is copied
    """
    if len(A.shape) != 3:
        raise ValueError('A must be a 3D tensor.')
    if copy:
        A = A.copy()

    U = np.eye(A.shape[1], dtype=A.dtype)
    for it in range(n_iter):
        acc = jacobi_rotations(A, U)
        delta = off_diagonal_measure(A)
        if verbose:
            print('Iteration: {} Delta: {} Acc: {}'.format(it, delta, acc))
        if delta < eps:
            break

    return A, U