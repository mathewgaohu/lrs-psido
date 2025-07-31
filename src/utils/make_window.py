from typing import Sequence

import numpy as np
import scipy.signal.windows as win


def make_window(
    shape: Sequence[int],
    boundary: Sequence[int],
    ex: float = 6.0,
    ll: float = 1.5,
):
    """
    Args:
        shape: Shape of the output array.
        boundary: [left, right, up, down] Margins of the window.
        ex: Controls the decreasing speed.
        ll: Controls the decreasing speed.
    """
    nx, nz = shape
    nxl, nxr, nzu, nzd = boundary

    W = np.ones((nx, nz))
    zmin, xmin = nzu, nxl
    zmax, xmax = nz - nzd, nx - nxr
    W[:xmin, :] *= np.exp(-((-np.linspace(-ll, 0, nxl, endpoint=False)) ** ex)).reshape(
        -1, 1
    )
    W[xmax:, :] *= np.exp(
        -(-np.linspace(-ll, 0, nxr, endpoint=False))[::-1] ** ex
    ).reshape(-1, 1)
    W[:, :zmin] *= np.exp(-((-np.linspace(-ll, 0, nzu, endpoint=False)) ** ex)).reshape(
        1, -1
    )
    W[:, zmax:] *= np.exp(
        -(-np.linspace(-ll, 0, nzd, endpoint=False))[::-1] ** ex
    ).reshape(1, -1)

    return W


def tukey_window_2d(shape: Sequence[int], alpha: Sequence[float]):
    """
    Generate a 2D Tukey window that only reduces values near boundaries.

    Parameters:
        shape (tuple): Shape of the desired 2D window (rows, cols).
        alpha (tuple): Tapering parameter (0 = rectangular, 1 = Hann-like).

    Returns:
        np.ndarray: 2D Tukey window mask.
    """
    nx, ny = shape
    alpha_x, alpha_y = alpha
    win_x = win.tukey(nx, alpha_x)  # 1D Tukey window along x
    win_y = win.tukey(ny, alpha_y)  # 1D Tukey window along y

    # Create 2D window using outer product
    window_2d = np.outer(win_x, win_y)

    return window_2d
