import numpy as np

def idr_solver(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    IDR (Induced Dimension Reduction) method to solve Ax=b.

    Parameters:
        A (numpy.ndarray or scipy.sparse.linalg.LinearOperator): Coefficient matrix.
        b (numpy.ndarray): Right-hand side vector.
        x0 (numpy.ndarray, optional): Initial guess for the solution. Default is None.
        tol (float, optional): Tolerance for convergence. Default is 1e-6.
        max_iter (int, optional): Maximum number of iterations. Default is 100.

    Returns:
        x (numpy.ndarray): Solution vector.
        converged (bool): True if the method converged, False otherwise.
        num_iterations (int): Number of iterations performed.
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    n = len(b)
    V = np.zeros((n, max_iter+1))
    H = np.zeros((max_iter+1, max_iter))
    R = np.zeros((n, max_iter+1))

    x = x0.copy()
    r = b - np.dot(A, x)
    beta = np.linalg.norm(r)
    V[:, 0] = r / beta
    R[:, 0] = r

    for k in range(max_iter):
        w = np.dot(A, V[:, k])
        for j in range(k + 1):
            H[j, k] = np.dot(w, V[:, j])
            w = w - H[j, k] * V[:, j]

        H[k + 1, k] = np.linalg.norm(w)
        V[:, k + 1] = w / H[k + 1, k]
        G = np.linalg.solve(H[:k+1, :k+1], beta * np.eye(k + 1))
        x = x + np.dot(V[:, :k+1], G)
        r = b - np.dot(A, x)
        beta = np.linalg.norm(r)

        R[:, k + 1] = r

        # Check for convergence
        if beta < tol:
            return x, True, k + 1

    return x, False, max_iter

# Example Usage:
if __name__ == "__main__":
    # Example matrix A and vector b
    A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]])
    b = np.array([15, 10, 10, 10])

    # Solve Ax=b using IDR
    x, converged, num_iterations = idr_solver(A, b)

    # Print the results
    if converged:
        print("IDR converged to a solution in {} iterations.".format(num_iterations))
        print("Solution x:")
        print(x)
    else:
        print("IDR did not converge within the specified tolerance.")
