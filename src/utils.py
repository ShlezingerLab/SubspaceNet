"""Subspace-Net 
Details
----------
Name: utils.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose:
--------
This script defines some helpful functions:
    * sum_of_diag: returns the some of each diagonal in a given matrix.
    * sum_of_diag_torch: returns the some of each diagonal in a given matrix, Pytorch oriented.
    * find_roots: solves polynomial equation defines by polynomial coefficients. 
    * find_roots_torch: solves polynomial equation defines by polynomial coefficients, Pytorch oriented.. 
    * set_unified_seed: Sets unified seed for all random attributed in the simulation.
    * get_k_angles: Retrieves the top-k angles from a prediction tensor.
    * get_k_peaks: Retrieves the top-k peaks (angles) from a prediction tensor using peak finding.
    * gram_diagonal_overload(self, Kx: torch.Tensor, eps: float): generates Hermitian and PSD (Positive Semi-Definite) matrix,
        using gram operation and diagonal loading.
"""

# Imports
import numpy as np
import torch
import random
import scipy

# Constants
R2D = 180 / np.pi
D2R = 1 / R2D
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Functions
# def sum_of_diag(matrix: np.ndarray) -> list:
def sum_of_diag(matrix: np.ndarray):
    """Calculates the sum of diagonals in a square matrix.

    Args:
        matrix (np.ndarray): Square matrix for which diagonals need to be summed.

    Returns:
        list: A list containing the sums of all diagonals in the matrix, from left to right.

    Raises:
        None

    Examples:
        >>> matrix = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        >>> sum_of_diag(matrix)
        [7, 12, 15, 8, 3]

    """
    diag_sum = []
    diag_index = np.linspace(
        -matrix.shape[0] + 1,
        matrix.shape[0] + 1,
        2 * matrix.shape[0] - 1,
        endpoint=False,
        dtype=int,
    )
    for idx in diag_index:
        diag_sum.append(np.sum(matrix.diagonal(idx)))
    return diag_sum


def sum_of_diags_torch(matrix: torch.Tensor):
    """Calculates the sum of diagonals in a square matrix.
    equivalent sum_of_diag, but support Pytorch.

    Args:
        matrix (torch.Tensor): Square matrix for which diagonals need to be summed.

    Returns:
        torch.Tensor: A list containing the sums of all diagonals in the matrix, from left to right.

    Raises:
        None

    Examples:
        >>> matrix = torch.tensor([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]])
        >>> sum_of_diag(matrix)
            torch.tensor([7, 12, 15, 8, 3])
    """
    diag_sum = []
    diag_index = torch.linspace(
        -matrix.shape[0] + 1, matrix.shape[0] - 1, 2 * matrix.shape[0] - 1, dtype=int
    )
    for idx in diag_index:
        diag_sum.append(torch.sum(torch.diagonal(matrix, idx)))
    return torch.stack(diag_sum, dim=0)


# def find_roots(coefficients: list) -> np.ndarray:
def find_roots(coefficients: list):
    """Finds the roots of a polynomial defined by its coefficients.

    Args:
        coefficients (list): List of polynomial coefficients in descending order of powers.

    Returns:
        np.ndarray: An array containing the roots of the polynomial.

    Raises:
        None

    Examples:
        >>> coefficients = [1, -5, 6]  # x^2 - 5x + 6
        >>> find_roots(coefficients)
        array([3., 2.])

    """
    coefficients = np.array(coefficients)
    A = np.diag(np.ones((len(coefficients) - 2,), coefficients.dtype), -1)
    if np.abs(coefficients[0]) == 0:
        A[0, :] = -coefficients[1:] / (coefficients[0] + 1e-9)
    else:
        A[0, :] = -coefficients[1:] / coefficients[0]
    roots = np.array(np.linalg.eigvals(A))
    return roots


def find_roots_torch(coefficients: torch.Tensor):
    """Finds the roots of a polynomial defined by its coefficients.
    equivalent to src.utils.find_roots, but support Pytorch.

    Args:
        coefficients (torch.Tensor): List of polynomial coefficients in descending order of powers.

    Returns:
        torch.Tensor: An array containing the roots of the polynomial.

    Raises:
        None

    Examples:
        >>> coefficients = torch.tensor([1, -5, 6])  # x^2 - 5x + 6
        >>> find_roots(coefficients)
        tensor([3., 2.])

    """
    A = torch.diag(torch.ones(len(coefficients) - 2, dtype=coefficients.dtype), -1)
    A[0, :] = -coefficients[1:] / coefficients[0]
    roots = torch.linalg.eigvals(A)
    return roots


def set_unified_seed(seed: int = 42):
    """
    Sets the seed value for random number generators in Python libraries.

    Args:
        seed (int): The seed value to set for the random number generators. Defaults to 42.

    Returns:
        None

    Raises:
        None

    Examples:
        >>> set_unified_seed(42)

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# def get_k_angles(grid_size: float, k: int, prediction: torch.Tensor) -> torch.Tensor:
def get_k_angles(grid_size: float, k: int, prediction: torch.Tensor):
    """
    Retrieves the top-k angles from a prediction tensor.

    Args:
        grid_size (float): The size of the angle grid (range) in degrees.
        k (int): The number of top angles to retrieve.
        prediction (torch.Tensor): The prediction tensor containing angle probabilities, sizeof equal to grid_size .

    Returns:
        torch.Tensor: A tensor containing the top-k angles in degrees.

    Raises:
        None

    Examples:
        >>> grid_size = 6
        >>> k = 3
        >>> prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
        >>> get_k_angles(grid_size, k, prediction)
        tensor([ 90., -18.,   54.])

    """
    angles_grid = torch.linspace(-90, 90, grid_size)
    doa_prediction = angles_grid[torch.topk(prediction.flatten(), k).indices]
    return doa_prediction


# def get_k_peaks(grid_size, k: int, prediction) -> torch.Tensor:
def get_k_peaks(grid_size: int, k: int, prediction: torch.Tensor):
    """
    Retrieves the top-k peaks (angles) from a prediction tensor using peak finding.

    Args:
        grid_size (int): The size of the angle grid (range) in degrees.
        k (int): The number of top peaks (angles) to retrieve.
        prediction (torch.Tensor): The prediction tensor containing the peak values.

    Returns:
        torch.Tensor: A tensor containing the top-k angles in degrees.

    Raises:
        None

    Examples:
        >>> grid_size = 6
        >>> k = 3
        >>> prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
        >>> get_k_angles(grid_size, k, prediction)
        tensor([ 90., -18.,   54.])

    """
    angels_grid = torch.linspace(-90, 90, grid_size)
    peaks, peaks_data = scipy.signal.find_peaks(
        prediction.detach().numpy().flatten(), prominence=0.05, height=0.01
    )
    peaks = peaks[np.argsort(peaks_data["peak_heights"])[::-1]]
    doa_prediction = angels_grid[peaks]
    while doa_prediction.shape[0] < k:
        doa_prediction = torch.cat(
            (
                doa_prediction,
                torch.Tensor(np.round(np.random.rand(1) * 180, decimals=2) - 90.00),
            ),
            0,
        )

    return doa_prediction[:k]


# def gram_diagonal_overload(Kx: torch.Tensor, eps: float) -> torch.Tensor:
def gram_diagonal_overload(Kx: torch.Tensor, eps: float, batch_size: int):
    """Multiply a matrix Kx with its Hermitian conjecture (gram matrix),
        and adds eps to the diagonal values of the matrix,
        ensuring a Hermitian and PSD (Positive Semi-Definite) matrix.

    Args:
    -----
        Kx (torch.Tensor): Complex matrix with shape [BS, N, N],
            where BS is the batch size and N is the matrix size.
        eps (float): Constant multiplier added to each diagonal element.
        batch_size(int): The number of batches

    Returns:
    --------
        torch.Tensor: Hermitian and PSD matrix with shape [BS, N, N].

    """
    # Insuring Tensor input
    if not isinstance(Kx, torch.Tensor):
        Kx = torch.tensor(Kx)

    Kx_list = []
    bs_kx = Kx
    for iter in range(batch_size):
        K = bs_kx[iter]
        # Hermitian conjecture
        Kx_garm = torch.matmul(torch.t(torch.conj(K)), K).to(device)
        # Diagonal loading
        eps_addition = (eps * torch.diag(torch.ones(Kx_garm.shape[0]))).to(device)
        Rz = Kx_garm + eps_addition
        Kx_list.append(Rz)
    Kx_Out = torch.stack(Kx_list, dim=0)
    return Kx_Out


if __name__ == "__main__":
    # sum_of_diag example
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sum_of_diag(matrix)

    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sum_of_diags_torch(matrix)

    # find_roots example
    coefficients = [1, -5, 6]
    find_roots(coefficients)

    # get_k_angles example
    grid_size = 6
    k = 3
    prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
    get_k_angles(grid_size, k, prediction)

    # get_k_peaks example
    grid_size = 6
    k = 3
    prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
    get_k_peaks(grid_size, k, prediction)
