from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def create_interpolation_weights(
    n: Sequence[int],
    indices: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Create interpolation weights for a given region using Gaussian kernels.

    Parameters:
        n (Sequence[int]): Shape of the region as a sequence of integers.
        indices (np.ndarray): Array of shape (nsamples, ndim) representing the indices of sample points.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: A weight array of shape (nsamples, *n) where each slice along the first axis contains
            the interpolation weights for the corresponding sample point.

    Notes:
        - The weights are computed using a Gaussian kernel centered at each sample point.
        - The weights are normalized so that their sum across all sample points is 1 for each grid point.
    """
    assert len(n) == indices.shape[1]
    nsamples, ndim = indices.shape

    # Create a mesh grid for the region
    coords = [np.arange(n[k]) for k in range(ndim)]
    Coords = np.moveaxis(np.meshgrid(*coords, indexing="ij"), 0, -1)

    # Initialize weights array
    weights = np.zeros((len(indices), *n))

    for idx, index in enumerate(indices):
        # # # RBF approach (give negative weights, not reliable)
        # delta = np.zeros(nsamples)
        # delta[idx] = 1
        # rbf = RBFInterpolator(indices, delta)
        # weights[idx] = rbf(Coords.reshape(-1, ndim)).reshape(n)

        # Gaussian approach
        distances = np.linalg.norm(Coords - index, ord=2, axis=-1)
        weights[idx] = np.exp(-(distances**2) / sigma**2)

    # Normalize weights to sum to 1
    weights /= np.sum(weights, axis=0, keepdims=True)

    return weights


def test_create_interpolation_weights():
    n = (50, 40)
    indices = np.array([(x, y) for x in [10, 20, 30, 40] for y in [10, 20, 30]])
    weights = create_interpolation_weights(n, indices, sigma=5)
    plt.matshow(weights[3].T)
    plt.colorbar()
    plt.savefig("tmp.png")
    weights = create_interpolation_weights(n, indices, sigma=7.5)
    plt.matshow(weights[3].T)
    plt.colorbar()
    plt.savefig("tmp.png")
