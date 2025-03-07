# src/force_training_rate_network/utils.py
import numpy as np

def connectivity_matrix_internal(N, shape, p, RNG):
    """
    Generate internal connectivity matrix.

    Args:
        N (int): Number of neurons.
        shape (tuple): Shape of the matrix.
        p (float): Connection probability.
        RNG (numpy.random.Generator): Random number generator.

    Returns:
        numpy.ndarray: Connectivity matrix.
    """
    scale = 1.0 / np.sqrt(p * N)
    M = RNG.normal(size=shape)
    mask = RNG.random(shape) < p
    M *= mask
    M -= M.mean()
    return M * scale

def connectivity_matrix_input_to_network(N, num_inpt, RNG):
    """
    Generate connectivity matrix from input to network.

    Args:
        N (int): Number of neurons in the network.
        num_inpt (int): Number of input neurons.
        RNG (numpy.random.Generator): Random number generator.

    Returns:
        numpy.ndarray: Connectivity matrix.
    """
    matrix = np.zeros((N, num_inpt))
    for i in range(N):
        col_index = RNG.integers(num_inpt)
        matrix[i, col_index] = RNG.normal(0, 1)
    return matrix
