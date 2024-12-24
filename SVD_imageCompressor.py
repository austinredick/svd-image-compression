class ImgCompression(object):
    """
    Computes the Singular Value Decomposition (SVD) of the input image.

    - SVD decomposes a matrix `X` into three components:
      X = U * S * V^T
      where:
        U: Orthogonal matrix of left singular vectors (columns are eigenvectors of X * X^T)
        S: Diagonal matrix of singular values (square roots of eigenvalues of X * X^T or X^T * X)
        V^T: Orthogonal matrix of right singular vectors (rows are eigenvectors of X^T * X)

    - If the input is a color image (3D array), SVD is computed separately for each color channel (R, G, B).

    Parameters:
    - X: Input image (2D for grayscale, 3D for color)

    Returns:
    U, S, V: SVD components (stacked for color images)
    """


    def __init__(self):
        pass

    def svd(self, X):
        # look at dimension of X to determine if color or gray-scale
        if X.ndim == 3:
            # split into red, green, blue color channels and apply SVD to each channel
            b, g, r = X[:, :, 0], X[:, :, 1], X[:, :, 2]

            u_b, s_b, v_b = np.linalg.svd(b)
            u_g, s_g, v_g = np.linalg.svd(g)
            u_r, s_r, v_r = np.linalg.svd(r)

            # stack results for all channels
            U = np.dstack((u_b, u_g, u_r))
            S = np.dstack((s_b, s_g, s_r))
            V = np.dstack((v_b, v_g, v_r))
        else:
            U, S, V = np.linalg.svd(X)

        return U, S, V

    def rebuild_svd(self, U, S, V, k):
        if U.ndim == 3:
            # reconstruct each color channel
            Xrebuild = np.zeros((U.shape[0], V.shape[0], 3))
            for i in range(3):
                val = np.dot(U[:, :k, i], np.dot(np.diag(S[0, :k, i]), V[:k, :, i]))
                Xrebuild[:, :, i] = val
        else:
            Xrebuild = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))
        return Xrebuild

    def compression_ratio(self, X, k):
        # formula: (compressed data) / (original data)
        return (k * (X.shape[0] + X.shape[1]) + k) / (X.shape[0] * X.shape[1])

    def recovered_variance_proportion(self, S, k):
        # calculate proportion of variance retained by color channel
        if S.ndim == 3:
            recovered_var = np.zeros(3)
            for i in range(3):
                recovered_var[i] = np.sum(S[0, :k, i] ** 2) / np.sum(S[0, :, i] ** 2)
        else:
            recovered_var = np.sum(S[:k] ** 2) / np.sum(S ** 2)
        return recovered_var