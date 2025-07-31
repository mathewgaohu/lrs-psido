"""Approximate symbol rows and columns."""

from typing import Sequence

import numpy as np
from scipy import fft
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .schemes import get_scheme
from .utils import create_vector_by_indices


def approximate_symbol_columns(
    op: LinearOperator,
    n: Sequence[int],
    settings: list[dict],
    op_vecs: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Approximate symbol columns of a linear operator under give settings.

    Args:
        op (LinearOperator): The linear operator.
        n (Sequence[int]): A tuple (nx, ny) representing the grid sizes.
        settings (list[dict]): A list of settings for the method. Each dictionary specifies an action and contains:
            - "sampling" (dict): Sampling settings for creating the sampling scheme.
            - "splitting" (dict, optional): Splitting settings for creating the splitting scheme.
              If not provided, the same scheme as "sampling" will be used.

    Returns:
        tuple[np.ndarray, np.ndarray, list[dict]]:
            - cols (np.ndarray): Symbol columns computed via multiple actions, with shape (M, nx, ny).
            - indices (np.ndarray): Indices used to create probing vectors, with shape (M, 2).
            - info (list[dict]): Information for each action, where each dictionary contains:
                - indices (np.ndarray): Indices used to create one probing vector.
                - v (np.ndarray): Probing vector.
                - op_v (np.ndarray): The result of applying the operator to the probing vector.
                - cols (np.ndarray): Computed symbol columns for the action.
    """
    n = np.asarray(n)

    all_cols = []
    all_col_indices = []
    all_info = []
    for i, this_setting in enumerate(settings):
        info = {}

        # Sampling
        sampling_setting = this_setting["sampling"]
        scheme = get_scheme(n=n, fft_index=True, **sampling_setting)
        indices = scheme.create_index()

        # Create vector
        vec = create_vector_by_indices(indices, n, fft_index=True)

        # Apply vector
        if op_vecs is not None:
            op_vec = op_vecs[i].reshape(n)
        else:
            op_vec = (op.H * vec.reshape(-1)).reshape(n)

        # Convert to specturm
        op_vec_fft = fft.fft2(op_vec)

        # Spliting
        spliting_setting = this_setting.get("spliting", None)
        if spliting_setting is not None:
            scheme = get_scheme(n=n, fft_index=True, **spliting_setting)
        parts = scheme.split(op_vec_fft, indices)

        # Convert to symbol columns
        cols = fft.ifft2(parts)

        info = {
            "indices": indices,
            "v": vec,
            "op_v": op_vec,
            "cols": cols,
        }
        all_col_indices.append(indices)
        all_cols.append(cols)
        all_info.append(info)

    all_cols = np.concatenate(all_cols, axis=0)
    all_col_indices = np.concatenate(all_col_indices, axis=0)

    return all_cols, all_col_indices, all_info


def approximate_symbol_rows(
    op: LinearOperator,
    n: Sequence[int],
    settings: list[dict],
    op_vecs: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:

    n = np.asarray(n)

    all_rows = []
    all_row_indices = []
    all_info = []
    for i, this_setting in enumerate(settings):
        info = {}

        # Sampling
        sampling_setting = this_setting["sampling"]
        scheme = get_scheme(n=n, fft_index=False, **sampling_setting)
        indices = scheme.create_index()

        # Create vector
        vec = create_vector_by_indices(indices, n, fft_index=False)

        # Apply vector
        if op_vecs is not None:
            op_vec = op_vecs[i].reshape(n)
        else:
            op_vec = (op.H * vec.reshape(-1)).reshape(n)

        # Spliting
        spliting_setting = this_setting.get("spliting", None)
        if spliting_setting is not None:
            scheme = get_scheme(n=n, fft_index=False, **spliting_setting)
        parts = scheme.split(op_vec, indices)

        # Convert to symbol columns
        rows = np.conj(fft.fft2(parts))

        info = {
            "indices": indices,
            "v": vec,
            "op_v": op_vec,
            "rows": rows,
        }
        all_row_indices.append(indices)
        all_rows.append(rows)
        all_info.append(info)

    all_rows = np.concatenate(all_rows, axis=0)
    all_row_indices = np.concatenate(all_row_indices, axis=0)

    return all_rows, all_row_indices, all_info


def test_approximate_symbol_columns():
    n = (11, 11)
    N = n[0] * n[1]
    op = aslinearoperator(np.eye(N))
    settings = [
        {
            "sampling": {
                "method": "diamond",
                "steps": [2, 2],
                "constraint": {"l2_ratio": [0.0, 1.0]},
            }
        }
    ]
    cols, col_indices, info = approximate_symbol_columns(op, n, settings)
    assert np.allclose(cols, np.ones_like(cols))

    v = np.zeros(n, dtype=int)
    v[tuple(col_indices.T)] = 1
    print(fft.fftshift(v).T)


def test_approximate_symbol_rows():
    n = (11, 11)
    N = n[0] * n[1]
    op = aslinearoperator(np.eye(N))
    settings = [
        {
            "sampling": {
                "method": "equispace",
                "steps": [3, 3],
                "anchor": [5, 5],
                "constraint": {"x": [2, 8]},
            }
        }
    ]
    rows, row_indices, info = approximate_symbol_rows(op, n, settings)
    assert np.allclose(rows, np.ones_like(rows))

    v = np.zeros(n, dtype=int)
    v[tuple(row_indices.T)] = 1
    print(v.T)
