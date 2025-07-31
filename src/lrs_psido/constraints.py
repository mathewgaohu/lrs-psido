from typing import Callable

import numpy as np


def create_constraint(
    n: np.ndarray,
    fft_index: bool,
    **kwargs: dict[str, tuple[float, float]],
) -> Callable[[np.ndarray], bool]:
    """
    Creates a composite constraint function based on specified limits and keys.

    This function generates a composite constraint that checks if a given set of
    coordinates satisfies multiple conditions. The conditions are determined by
    the provided keyword arguments (`kwargs`).

    Args:
        n (np.ndarray): The shape of the grid (array dimensions).
        fft_index (bool): Indicates whether the indices correspond to FFT frequencies.
            This affects the behavior of the `l2_ratio` key.
        **kwargs: Arbitrary keyword arguments defining constraints. Supported keys are:
            - "x": Constraint along the first axis.
            - "y": Constraint along the second axis.
            - "l2_ratio": Constraint based on the normalized L2 norm of the coordinates.
                Requires `fft_index=True`.
            Each key maps to a tuple `(min_value, max_value)` defining the allowed range.

    Returns:
        Callable[[np.ndarray], bool]: A function that takes an array of coordinates
        as input and returns `True` if all constraints are satisfied, `False` otherwise.

    Example:
        >>> n = np.array([128, 128])
        >>> constraint = create_constraint(
        ...     n,
        ...     fft_index=True,
        ...     x=(0, 64),
        ...     l2_ratio=(0.0, 1.0)
        ... )
        >>> coords = np.array([10, 20])
        >>> constraint(coords)
        True
    """
    functions = []
    vmins = []
    vmaxs = []
    for key, limit in kwargs.items():
        vmins.append(limit[0])
        vmaxs.append(limit[1])
        if key == "x":
            functions.append(lambda coords: coords[..., 0])
        elif key == "y":
            functions.append(lambda coords: coords[..., 1])
        elif key == "l2_ratio":
            assert fft_index, "`l2_ratio` requires `fft_index=True`."
            functions.append(
                lambda coords: np.linalg.norm(coords / (n // 2), ord=2, axis=-1)
            )
        else:
            raise ValueError(f"Unkown key: {key}")

    func = lambda coords: np.array([f(coords) for f in functions])
    vmins = np.asarray(vmins)
    vmaxs = np.asarray(vmaxs)

    def constraint(coords: np.ndarray) -> bool:
        """
        Checks if the input coordinates satisfy all constraints.

        Args:
            coords (np.ndarray): Array of coordinates to check.

        Returns:
            bool: True if all constraints are satisfied, False otherwise.
        """
        out = func(coords)
        return np.all(out >= vmins) and np.all(out <= vmaxs)

    return constraint


# Tests
def test_constraint():
    n = np.array([10, 6])
    constraint = create_constraint(n, fft_index=True, l2_ratio=[0.1, 0.5])
    assert constraint(np.array([1, 1]))
    assert not constraint(np.array([0, 0]))
    assert not constraint(np.array([3, 0]))
    constraint = create_constraint(n, fft_index=True)
    assert constraint(np.array([0, 0]))

    n = np.array([461, 121])
    constraint = create_constraint(n, fft_index=False, x=[30, 430], y=[20, 100])
    assert constraint(np.array([30, 40]))
    assert not constraint(np.array([20, 40]))
