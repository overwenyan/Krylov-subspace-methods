import numpy as np
from gmres_solver import gmres_solver

def restarting_gmres_solver(A, b, restart, tol=1e-6, max_iter=100):
    """
    Restarting GMRES method to solve Ax=b.

    Parameters:
        A (numpy.ndarray or scipy.sparse.linalg.LinearOperator): Coefficient matrix.
        b (numpy.ndarray): Right-hand side vector.
        restart (int): Number of iterations before restarting the GMRES process.
        tol (float, optional): Tolerance for convergence. Default is 1e-6.
        max_iter (int, optional): Maximum number of iterations. Default is 100.

    Returns:
        x (numpy.ndarray): Solution vector.
        converged (bool): True if the method converged, False otherwise.
        num_iterations (int): Number of iterations performed.
    """
    n = len(b)
    x = np.zeros_like(b)
    r = b - np.dot(A, x)
    norm_b = np.linalg.norm(b)
    tol *= norm_b

    for k in range(max_iter):
        # Run GMRES for 'restart' iterations
        x, converged, _ = gmres_solver(A, b, x0=x, tol=tol, max_iter=restart)
        
        if converged:
            break

    return x, converged, k

# Example Usage:
if __name__ == "__main__":
    # Example matrix A and vector b
    A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]])
    b = np.array([15, 10, 10, 10])

    # Specify the number of iterations before restarting
    restart = 5

    # Solve Ax=b using Restarting GMRES
    x, converged, num_iterations = restarting_gmres_solver(A, b, restart)

    # Print the results
    if converged:
        print("Restarting GMRES converged to a solution in {} iterations.".format(num_iterations))
        print("Solution x:")
        print(x)
    else:
        print("Restarting GMRES did not converge within the specified tolerance.")
