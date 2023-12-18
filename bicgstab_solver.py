import numpy as np

def bicgstab_solver(A, b, x0=None, tol=1e-3, max_iter=100):
    """
    BiCGSTAB (Biconjugate Gradient Stabilized) method to solve Ax=b.

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

    r = b - np.dot(A, x0)
    r0 = r
    p = r.copy()
    rho_0 = alpha = omega = 1.0
    v = np.zeros_like(b)

    for k in range(max_iter):
        rho_1 = np.dot(r, r0)
        beta = (rho_1 / rho_0) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = np.dot(A, p)
        alpha = rho_1 / np.dot(r, v)
        h = x0 + alpha * p
        s = r - alpha * v
        t = np.dot(A, s)
        omega = np.dot(t, s) / np.dot(t, t)
        x = h + omega * s

        # Check for convergence
        if np.linalg.norm(r) < tol:
            return x, True, k + 1

        r = s - omega * t

        rho_0 = rho_1

    return x, False, max_iter

# Example Usage:
if __name__ == "__main__":
    # Example matrix A and vector b
    A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]])
    b = np.array([15, 10, 10, 10])

    # Solve Ax=b using BiCGSTAB
    x, converged, num_iterations = bicgstab_solver(A, b)

    # Print the results
    if converged:
        print("BiCGSTAB converged to a solution in {} iterations.".format(num_iterations))
        print("Solution x:")
        print(x)
    else:
        print("BiCGSTAB did not converge within the specified tolerance.")
