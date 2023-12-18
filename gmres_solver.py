import numpy as np
from scipy.sparse.linalg import gmres

def gmres_solver(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    GMRES method to solve Ax=b.

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

    def callback(xk):
        # Optional: You can add a callback function for additional monitoring.
        pass

    x, info = gmres(A, b, x0=x0, tol=tol, maxiter=max_iter, callback=callback)
    
    converged = (info == 0)
    num_iterations = len(x)

    return x, converged, num_iterations

# Example Usage:
if __name__ == "__main__":
    # Example matrix A and vector b
    A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]])
    b = np.array([15, 10, 10, 10])

    # Solve Ax=b using GMRES
    x, converged, num_iterations = gmres_solver(A, b)

    # Print the results
    if converged:
        print("GMRES converged to a solution in {} iterations.".format(num_iterations))
        print("Solution x:")
        print(x)
    else:
        print("GMRES did not converge within the specified tolerance.")
