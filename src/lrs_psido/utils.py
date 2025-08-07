import logging
from itertools import product
from typing import Callable

import numpy as np
from numpy import fft

logger = logging.getLogger(__name__)


def create_equispaced_array(
    anchor: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    """Creates a multi-dimensional equispaced integer array within ranges, containing an anchor point.

    Args:
        anchor (np.ndarray): Array of integers specifying the anchor in each dimension.
        min_vals (np.ndarray): Array of integers specifying the minimum value for each dimension.
        max_vals (np.ndarray): Array of integers specifying the maximum value for each dimension.
        steps (np.ndarray): Array of integers specifying the step size for each dimension.

    Returns:
        np.ndarray: An array of shape `(N, len(anchor))` where `N` is the number of equispaced points,
                    and each row corresponds to a point in the multi-dimensional grid.
    """
    if not (len(anchor) == len(min_vals) == len(max_vals) == len(steps)):
        raise ValueError(
            "All input arrays (anchor, min_vals, max_vals, steps) must have the same length."
        )

    # Validate that anchors are within their respective ranges
    if not np.all((min_vals <= anchor) & (anchor <= max_vals)):
        raise ValueError("Each anchor must be within its respective range.")

    # Generate equispaced values for each dimension
    grids = []
    for a, min_val, max_val, step in zip(anchor, min_vals, max_vals, steps):
        lower_part = np.arange(a, min_val - 1, -step)[::-1]
        upper_part = np.arange(a, max_val + 1, step)
        grid = np.unique(np.concatenate((lower_part, upper_part)))
        grids.append(grid)

    # Compute the Cartesian product of all grids
    cartesian_product = np.array(list(product(*grids)))

    return cartesian_product


def create_vector_by_indices(
    indices: np.ndarray, n: np.ndarray, fft_index: bool = False
) -> np.ndarray:
    """Create a probing vector (in spatial or frequency space) with given indices.

    Args:
        indices (np.ndarray): Array of shape (m, d), representing indices in frequency space.
        n (np.ndarray): Sequence of length d, specifying the shape of the probing vector.
        fft_index(bool): If the indices are in frequecy space.

    Returns:
        np.ndarray: The vector as a numpy array.
    """
    if fft_index:
        specturm = np.zeros(n)
        specturm[tuple(indices.T)] = 1.0
        v = fft.ifft2(specturm, norm="forward")
        if np.abs(v.imag).max() < 0.1 * np.abs(v.real).max():
            return v.real  # Imaginary part ignored.
        else:
            logger.info(
                "The imaginary part of the probing vector isn't ignorable. "
                + "The probing vector is complex. "
                + f"If it is not expected, check indices: {indices}"
            )
            return v
    else:
        v = np.zeros(n)
        v[tuple(indices.T)] = 1.0
        return v


def truncate_smoothly(
    x: np.ndarray,
    start: float,
    end: float,
    alpha: float = 3.0,
    order: float = 2.0,
    tmax: float = 1.0,
) -> np.ndarray:
    """Smooth truncation function as exp(alpha * t**order).

    This function smoothly decreases from 1 (at `start`) to 0 (at `end`) using an exponential
    decay formula. Values beyond `end` are set to 0. Values before `start` remain at 1
    unless `start == end`.

    Args:
        x (np.ndarray): The input array of values to be truncated.
        start (float): The starting point where the function value begins to decrease.
        end (float): The endpoint where the function value reaches 0.
        alpha (float, optional): The steepness parameter for the exponential decay.
            Defaults to 3.0.
        order (float, optional): The order of the decay, controlling how steeply it
            decreases. Defaults to 2.0.
        tmax (float, optional): The maximum value for the scaled parameter `t` used
            in the exponential formula. Defaults to 1.0.

    Returns:
        np.ndarray: An array with the same shape as `x`, where values are smoothly
        truncated based on the specified parameters.

    Raises:
        ValueError: If `start` is greater than `end`.

    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> y = exp_truncation(x, start=2, end=8)
    """
    if start > end:
        raise ValueError("`start` must not be greater than `end`.")

    if start == end:
        # When start and end are the same, output 1 for values <= start, else 0.
        return np.where(x <= start, 1.0, 0.0)

    # Scale and clip x values into the range [0, tmax].
    t = np.clip((x - start) / (end - start), 0, tmax)

    # Apply exponential truncation.
    out = np.exp(-alpha * t**order)

    # Explicitly set values beyond `end` to 0.
    out[x > end] = 0.0

    return out


def split(
    x: np.ndarray,
    indices: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    eps: float = 0.2,
    fft_index: bool = False,
    roll_to_origin: bool = True,
    alpha: float = 3.0,
    order: float = 2.0,
) -> np.ndarray:
    """
    Splits an input array into segments based on a smooth mask generated from a function.

    This function creates a smooth mask for each index in `indices` using the provided
    `func`, applies the mask to the input array `x`, and shifts each masked portion to
    the origin. The resulting segments are returned as an array.

    Args:
        x (np.ndarray): The input array to split.
        indices (np.ndarray): An array of shape `(num_indices, ndim)` specifying the
            indices for splitting. Each index corresponds to a location in the array.
        func (Callable[[np.ndarray], np.ndarray]): A function that generates the mask.
            It takes as input a shifted coordinate array (`Coords - index`).
        eps (float, optional): A smoothing parameter that adjusts the truncation threshold.
            Defaults to 0.1.
        fft_index (bool, optional): If True, the function uses FFT frequencies instead of
            regular indices for coordinate computation. Defaults to False.
        alpha (float): Input for truncate_smoothly().
        order (float): Input for truncate_smoothly().

    Returns:
        np.ndarray: An array of shape `(num_indices, *x.shape)` where each slice
        corresponds to a masked and shifted segment of the input array.

    Raises:
        AssertionError: If the number of dimensions in `x` does not match `indices.shape[1]`.

    Example:
        >>> x = np.random.rand(4, 4)
        >>> indices = np.array([[0, 0], [2, 2]])
        >>> def mask_func(coords):
        ...     return np.linalg.norm(coords, axis=-1)
        >>> result = split(x, indices, mask_func, eps=0.1, fft_index=False)
        >>> result.shape
        (2, 4, 4)
    """
    assert (
        len(x.shape) == indices.shape[1]
    ), "Dimensions of `x` and `indices` don't match."

    n = np.asarray(x.shape)
    d = len(n)

    # Generate coordinate grids based on whether FFT indices are used.
    if fft_index:
        coords = [fft.fftfreq(n[k], 1 / n[k]).astype(int) for k in range(d)]
        indices = np.stack([coords[k][indices[:, k]] for k in range(d)], axis=1)
    else:
        coords = [np.arange(n[k]) for k in range(d)]

    # Combine coordinate grids into a single array.
    Coords = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)  # Shape: (*n, d)

    outs = []
    for index in indices:
        # Create a mask using the provided function and truncation parameters.
        mask = truncate_smoothly(
            func(Coords - index), 1.0 - eps, 1.0, alpha=alpha, order=order
        )

        # Apply the mask to the input array.
        out = x * mask

        # Shift the masked portion to the origin.
        if roll_to_origin:
            out = np.roll(out, -index, axis=range(d))

        # Store the result.
        outs.append(out)

    # Return all processed segments as a single array.
    return np.asarray(outs)


# Tests
def test_create_equispaced_array():
    anchor = np.array([0])
    min_vals = np.array([-5])
    max_vals = np.array([5])
    steps = np.array([2])
    out = create_equispaced_array(anchor, min_vals, max_vals, steps)
    out_true = np.array([[-4], [-2], [0], [2], [4]])
    assert np.all(out == out_true), out

    anchor = np.array([1, 0])
    min_vals = np.array([-2, -3])
    max_vals = np.array([4, 2])
    steps = np.array([2, 3])
    out = create_equispaced_array(anchor, min_vals, max_vals, steps)
    out_true = np.array([[-1, -3], [-1, 0], [1, -3], [1, 0], [3, -3], [3, 0]])
    assert np.all(out == out_true), out


def test_split():
    x = np.arange(25).reshape(5, 5)
    indices = np.array([[1, 1], [3, 3]])
    func = lambda idx: np.linalg.norm(idx, ord=np.inf, axis=-1)
    eps = 0.1
    out = split(x, indices, func, eps).round().astype(int)
    out_true = np.array(
        [
            [
                [6, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [18, 1, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1],
            ],
        ]
    )
    assert np.all(out == out_true)
