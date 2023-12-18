import numpy as np

def bicg_solver(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    BiCG (Biconjugate Gradient) method to solve Ax=b.

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
    r_tilde = r.copy()
    p = r.copy()
    p_tilde = r_tilde.copy()
    rho = np.dot(r_tilde, r)

    for k in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rho / np.dot(p_tilde, Ap)
        x0 = x0 + alpha * p
        r = r - alpha * Ap
        r_tilde = r_tilde - alpha * np.dot(A.T, p_tilde)

        rho_new = np.dot(r_tilde, r)
        beta = rho_new / rho
        p = r + beta * p
        p_tilde = r_tilde + beta * p_tilde

        rho = rho_new

        # Check for convergence
        if np.linalg.norm(r) < tol:
            return x0, True, k + 1

    return x0, False, max_iter

# Example Usage:
if __name__ == "__main__":
    # Example matrix A and vector b
    A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]])
    b = np.array([15, 10, 10, 10])

    # Solve Ax=b using BiCG
    x, converged, num_iterations = bicg_solver(A, b)

    # Print the results
    if converged:
        print("BiCG converged to a solution in {} iterations.".format(num_iterations))
        print("Solution x:")
        print(x)
    else:
        print("BiCG did not converge within the specified tolerance.")
