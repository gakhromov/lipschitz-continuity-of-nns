import torch

# Optimised matrix norm calculations


@torch.jit.script
def compute_matrix_2norm_power_method(
    A: torch.Tensor, atol: float = 1e-3, max_iters: int = 100
) -> torch.Tensor:
    """This function finds the matrix 2-norm by computing the largest singular 
    value using the power method.
    This function works faster than the torch and numpy built-in 2-norm 
    computation.

    Parameters
    ----------
    A
        Matrix with dimensions (n x m).
    atol, optional
        Smallest absolute difference between two iterations, after which stop 
        the computation. By default 1e-3.
    max_iters, optional
        Maximum number of iterations allowed in the power method, by default 
        100.

    Returns
    -------
        The value of the norm.
    """
    torch.manual_seed(42)

    # get the Hermitian matrix
    if A.shape[0] > A.shape[1]:
        H = A.T @ A
    else:
        H = A @ A.T

    # get a normalised vector that we will use to compute the eigen value
    x = torch.rand(H.shape[0]).to(A.device)
    x = x / torch.linalg.norm(x, 2)

    lambda_estimate = torch.linalg.norm(H @ x, 2)
    if not lambda_estimate.is_nonzero():
        # if the lambda estimate is zero, that means that H @ x is zero.
        # if H @ x is zero, it means that either is x is randomly initialised in
        # such a way that it yields a zero vector (which is unlikely) or H is
        # all zeros.
        # Here, we check that H is not all zeros:
        if H.count_nonzero() == 0:
            # In this case we just output zero
            return torch.tensor(0).to(A.device)

        # Here, we regenerate x to check the unlikely scenario:
        x = torch.rand(H.shape[0]).to(A.device)
        x = x / torch.linalg.norm(x, 2)
        lambda_estimate = torch.linalg.norm(H @ x, 2)

        if not lambda_estimate.is_nonzero():
            # if it is still zero, than the H matrix must be all "almost" zeros,
            # and in this case we just output zero as well
            return torch.tensor(0).to(A.device)

    # largest singular value is the square root of the largest eigenvalue of the
    # Hermitian matrix
    sigma_max_estimate_prev = torch.sqrt(lambda_estimate)
    # this line is required for torch to compile the script
    sigma_max_estimate_next = sigma_max_estimate_prev

    for _ in range(max_iters):
        t = H @ x
        x = t / torch.linalg.norm(t, 2)
        sigma_max_estimate_next = torch.sqrt(torch.linalg.norm(H @ x, 2))
        if torch.abs(sigma_max_estimate_next - sigma_max_estimate_prev) <= atol:
            break
        sigma_max_estimate_prev = sigma_max_estimate_next

    return sigma_max_estimate_next


@torch.jit.script
def compute_matrix_2norm_power_method_batched(
    A: torch.Tensor, eps: float = 1e-3, max_iters: int = 100
) -> torch.Tensor:
    """This function finds the 2-norm of a batch of matrices
    by computing the largest singular value using the power method.

    Parameters
    ----------
    A
        Matrix with dimensions (batch_size x n x m).
    eps, optional
        Smallest difference between two iterations, after which stop the 
        computation. By default 1e-3.
    max_iters, optional
        Maximum number of iterations allowed in the power method, by default 
        100.

    Returns
    -------
        The values of the norm, in a form of a torch vector.
    """
    norms = [compute_matrix_2norm_power_method(
        A[i], eps, max_iters) for i in range(A.shape[0])]
    return torch.stack(norms)


@torch.jit.script
def compute_matrix_1norm(A: torch.Tensor) -> torch.Tensor:
    """This function finds the matrix 1-norm. It works faster than the standard 
    torch 1-norm.

    Parameters
    ----------
    A
        Matrix with dimensions (n x m).

    Returns
    -------
        The value of the norm.
    """
    column_sums = torch.sum(torch.abs(A), dim=0)
    return torch.max(column_sums)


@torch.jit.script
def compute_matrix_1norm_batched(A: torch.Tensor) -> torch.Tensor:
    """This function finds the 1-norm of a batch of matrices. It works faster 
    than the standard torch 1-norm.

    Parameters
    ----------
    A
        Matrix with dimensions (batch_size x n x m).
    Returns
    -------
        The values of the norm, in a form of a torch vector.
    """
    column_sums = torch.sum(torch.abs(A), dim=1)
    return torch.max(column_sums, dim=1).values
