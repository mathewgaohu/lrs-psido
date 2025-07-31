from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scila
from pdo.utils.weights import make_weights_polar_array_flat
from scipy import fft, ndimage
from scipy.sparse.linalg import LinearOperator

from .approx import approximate_symbol_columns, approximate_symbol_rows
from .filters import GaussianHighPassFilter
from .linear_operators import RealPDOLowRankSymbol
from .plot import (
    plot_pdo_process,
    plot_psf_process,
    plot_psido,
    plot_vec,
    plot_vec_with_fft,
)
from .weights import create_interpolation_weights


class Approximator:

    sqrt: bool  # Whether approximate sqrt(A) instead of A

    def _approximate(self, A: LinearOperator) -> LinearOperator:
        """Implementation of approximating sqrt(A) or A."""
        raise NotImplementedError

    def approximate_sqrt(self, A: LinearOperator) -> LinearOperator:
        if self.sqrt:
            return self._approximate(A)
        else:
            raise NotImplementedError

    def approximate(self, A: LinearOperator) -> LinearOperator:
        if self.sqrt:
            Ahat = self.approximate_sqrt(A)
            return Ahat @ Ahat.H
        else:
            return self._approximate(A)

    def __call__(self, A: LinearOperator) -> LinearOperator:
        return self.approximate(A)


@dataclass
class PdoApproximator(Approximator):
    """Approximates a linear operator using low-rank symbol approximation with PDO method."""

    # Define specifications in order of process
    n: np.ndarray  # Size of Domain
    filter_sigma: float = 0.0  # Sigma of high-pass filter
    matvec_plan: list[dict] = None  # List of configuration for each matvec.
    l2_ratio: float = None
    num_angles: int = None
    smooth: bool = False  # Smoothen computed symbol
    real_only: bool = False  # Restrict to real symbol
    sqrt: bool = True  # Use square root scaling
    rescale: bool = False  # Rescale the weights, NOT availble
    log_dir: str = None  # Directory for logging results
    filter_mode: str = "both"  # "both", "left", "right"
    show_proc: bool = False

    def __post_init__(self):
        self.n = np.asarray(self.n, dtype=int)
        self.log_dir = Path(self.log_dir) if self.log_dir else None
        self.ncalls = 0

        if self.filter_sigma > 0:
            self.Fh = GaussianHighPassFilter(self.n, self.filter_sigma)
        else:
            self.Fh = None

        # Prepare interpolation weights
        if self.l2_ratio is None:
            self.l2_ratio = self.matvec_plan[0]["sampling"]["l2_ratio"]
        if self.num_angles is None:
            self.num_angles = self.matvec_plan[0]["sampling"]["num_angles"]
        n_probe = self.num_angles // 2
        angles = np.linspace(0, 2 * np.pi, 2 * n_probe, endpoint=False)
        weights = make_weights_polar_array_flat(
            None, angles, None, None, L=self.n - 1, n=self.n
        )
        weights = np.asarray(weights).reshape(-1, *self.n)

        nx, ny = self.n
        xxf = fft.fftfreq(nx, 1) * 2 * nx / (nx - 1)
        yyf = fft.fftfreq(ny, 1) * 2 * ny / (ny - 1)
        XXf, YYf = np.meshgrid(xxf, yyf, indexing="ij")
        RRf = np.sqrt(XXf**2 + YYf**2)

        # Apply scaling
        scaling_factor = np.sqrt(RRf / self.l2_ratio) if self.sqrt else (RRf / self.l2_ratio)
        weights *= scaling_factor

        # Construct PDO row components
        self.n_probe = n_probe
        self.pdo_r = np.vstack(
            (
                weights[:n_probe] + weights[n_probe:],
                1j * weights[:n_probe] - 1j * weights[n_probe:],
            )
        )

        if self.real_only:
            self.pdo_r = self.pdo_r[: self.n_probe]

    def _approximate(self, A: LinearOperator) -> RealPDOLowRankSymbol:
        """Approximates A or sqrt(A) with PDO method

        Args:
            A: The input linear operator.

        Returns:
            RealPDOLowRankSymbol: The Approximation
        """
        self.ncalls += 1

        if self.Fh:
            if self.filter_mode == "both":
                A = self.Fh @ A @ self.Fh.H
            elif self.filter_mode == "left":
                A = self.Fh @ A
            elif self.filter_mode == "right":
                A = A @ self.Fh.H
            else:
                raise ValueError(self.filter_mode)

        # Compute approximate symbol columns
        cols, col_indices, info = approximate_symbol_columns(
            A, self.n, self.matvec_plan
        )

        # Smooth columns if needed
        if self.smooth:
            cols = smoothen(cols, sigma=2, fft_index=False)

        # Apply square root transformation if needed
        if self.sqrt:
            cols = np.sqrt(cols)

        # Construct PDO column components
        pdo_c = np.vstack((cols[: self.n_probe].real, cols[: self.n_probe].imag))

        if self.real_only:
            pdo_c = pdo_c[: self.n_probe]

        # Log results if directory is specified
        if self.log_dir:
            path = self.log_dir / f"approx_{self.ncalls}"
            save_pdo(path, pdo_c, self.pdo_r)
            save_symbol_cols(path, cols, col_indices, info, show_proc=self.show_proc)

        Ahat = RealPDOLowRankSymbol(pdo_c, self.pdo_r)

        return Ahat


@dataclass
class PsfApproximator(Approximator):
    """Approximates a linear operator using low-rank symbol approximation with PSF method."""

    # Define specifications in order of process
    n: np.ndarray  # Size of Domain
    filter_sigma: float = 0.0  # Sigma of high-pass filter
    matvec_plan: list[dict] = None  # List of configuration for each matvec.
    smooth: bool = False  # Smoothen computed symbol
    real_only: bool = False  # Restrict to real symbol
    sqrt: bool = True  # Use square root scaling
    rescale: bool = False  # Rescale the weights, NOT availble
    log_dir: str = None  # Directory for logging results
    window: np.ndarray = field(default=None, repr=False)
    filter_mode: str = "both"  # "both", "left", "right"
    show_proc: bool = False

    def __post_init__(self):
        self.n = np.asarray(self.n, dtype=int)
        self.log_dir = Path(self.log_dir) if self.log_dir else None
        self.ncalls = 0

        if self.filter_sigma > 0:
            self.Fh = GaussianHighPassFilter(self.n, self.filter_sigma)
        else:
            self.Fh = None

    def _approximate(self, A: LinearOperator) -> RealPDOLowRankSymbol:
        """Approximates A or sqrt(A) with PSF method

        Args:
            A: The input linear operator.

        Returns:
            RealPDOLowRankSymbol: The Approximation
        """
        self.ncalls += 1

        if self.Fh:
            if self.filter_mode == "both":
                A = self.Fh @ A @ self.Fh.H
            elif self.filter_mode == "left":
                A = self.Fh @ A
            elif self.filter_mode == "right":
                A = A @ self.Fh.H
            else:
                raise ValueError(self.filter_mode)

        # Approximate symbol rows
        rows, row_indices, info = approximate_symbol_rows(A, self.n, self.matvec_plan)

        # Sort sample points
        iisort = np.lexsort(row_indices[:, ::-1].T)
        rows = rows[iisort]
        row_indices = row_indices[iisort]

        # Create interpolation weights
        sigma = 0.75 * np.linalg.norm(row_indices[1] - row_indices[0])
        weights = create_interpolation_weights(self.n, row_indices, sigma=sigma)

        if self.smooth:
            rows = smoothen(rows, sigma=self.n / self.n[1] * 2, fft_index=True)

        if self.real_only:
            rows = rows.real * (rows.real > 0)
        else:
            rows[:, 0, 0][rows[:, 0, 0].real < 0] = 0

        if self.sqrt:
            rows = np.sqrt(rows)

        nsamples = len(row_indices)
        pdo_c = weights * self.window
        pdo_r = rows

        # Log results if directory is specified
        if self.log_dir:
            path = self.log_dir / f"approx_{self.ncalls}"
            save_pdo(path, pdo_c, pdo_r)
            save_symbol_rows(path, rows, row_indices, info, show_proc=self.show_proc)

        Ahat = RealPDOLowRankSymbol(pdo_c, pdo_r)

        return Ahat


@dataclass
class PsfPlusApproximator(Approximator):
    """Approximates a linear operator using low-rank symbol approximation with PSF+ method."""

    # Define specifications in order of process
    n: np.ndarray  # Size of Domain
    filter_sigma: float = 0.0  # Sigma of high-pass filter
    matvec_plan_rows: list[dict] = None  # Config for each matvec for symbol rows.
    matvec_plan_cols: list[dict] = None  # Config for each matvec for symbol cols.
    smooth_rows: bool = False  # Smoothen computed symbol rows
    smooth_cols: bool = False  # Smoothen computed symbol cols
    real_only: bool = False  # Restrict to real symbol
    sqrt: bool = True  # Use square root scaling
    rescale: bool = False  # Rescale the weights, NOT availble
    symbol_rank: int = 10  # Target symbol rank
    log_dir: str = None  # Directory for logging results
    window: np.ndarray = field(default=None, repr=False)
    filter_mode: str = "both"  # "both", "left", "right"
    show_proc: bool = False

    def __post_init__(self):
        self.n = np.asarray(self.n, dtype=int)
        self.log_dir = Path(self.log_dir) if self.log_dir else None
        self.ncalls = 0

        if self.filter_sigma > 0:
            self.Fh = GaussianHighPassFilter(self.n, self.filter_sigma)
        else:
            self.Fh = None

    def _approximate(self, A: LinearOperator) -> RealPDOLowRankSymbol:
        """Approximates A or sqrt(A) with PSF+ method

        Args:
            A: The input linear operator.

        Returns:
            RealPDOLowRankSymbol: The Approximation
        """
        self.ncalls += 1

        if self.Fh:
            if self.filter_mode == "both":
                A = self.Fh @ A @ self.Fh.H
            elif self.filter_mode == "left":
                A = self.Fh @ A
            elif self.filter_mode == "right":
                A = A @ self.Fh.H
            else:
                raise ValueError(self.filter_mode)

        n = self.n
        N = np.prod(n)

        # Compute symbol rows
        rows, row_indices, row_info = approximate_symbol_rows(
            A, n, self.matvec_plan_rows
        )
        print(f"{rows.shape = }")
        # Rearange rows
        iisort = np.lexsort(row_indices[:, ::-1].T)
        rows = rows[iisort]
        row_indices = row_indices[iisort]

        cols, col_indices, col_info = approximate_symbol_columns(
            A, n, self.matvec_plan_cols
        )
        print(f"{cols.shape = }")

        # Backup rows and columns
        origin_rows = rows.copy()
        origin_cols = cols.copy()

        # Adjust rows and columns
        if self.smooth_rows:
            rows = smoothen(rows, sigma=n / n[1] * 2, fft_index=True)
        if self.smooth_cols:
            cols = smoothen(cols, sigma=2, fft_index=False)

        if self.real_only:
            rows = rows.real * (rows.real > 0)
            cols = cols.real * (cols.real > 0)
        else:
            rows[:, 0, 0][rows[:, 0, 0].real < 0] = 0

        if self.sqrt:
            rows = np.sqrt(rows)
            cols = np.sqrt(cols)

        # Compute low-rank symbol
        pdo_c, pdo_r = compute_pdo_xq(
            cols,
            rows,
            col_indices,
            row_indices,
            window=self.window,
            rank=self.symbol_rank,
            rescacle_rows=self.rescale,
        )

        # Log results if directory is specified
        if self.log_dir:
            print("Saving computed symbol")
            path = self.log_dir / f"approx_{self.ncalls}"
            save_pdo(path, pdo_c, pdo_r)
            save_symbol_rows(
                path, rows, row_indices, row_info, show_proc=self.show_proc
            )
            save_symbol_cols(
                path, cols, col_indices, col_info, show_proc=self.show_proc
            )

        Ahat = RealPDOLowRankSymbol(pdo_c, pdo_r)

        return Ahat


def smoothen(arrays: np.ndarray, sigma, fft_index: bool) -> np.ndarray:
    assert arrays.ndim == 3
    output = np.zeros_like(arrays)
    for i in range(arrays.shape[0]):
        x = arrays[i]
        if fft_index:
            x = fft.fftshift(x)
        x = ndimage.gaussian_filter(x, sigma=sigma, mode="nearest")
        if fft_index:
            x = fft.ifftshift(x)
        output[i] = x
    return output


def compute_pdo_xq(
    cols: np.ndarray,
    rows: np.ndarray,
    col_indices: np.ndarray,
    row_indices: np.ndarray,
    window: np.ndarray,
    rank: int,
    rescacle_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create low-rank symbol from columns and rows.

    TODO: better method for rows/cols -> LR symbol:
        - fourier basis for L and R. (not pure matrix completion)

    Note: May require more rows/cols? (more than true rank.)
    """
    n = np.asarray(cols.shape[1:])

    # Convert to matrix format.
    Cols = cols.reshape(-1, np.prod(n)).T
    Rows = rows.reshape(-1, np.prod(n))
    ic = np.arange(np.prod(n)).reshape(n)[tuple(col_indices.T)]
    ir = np.arange(np.prod(n)).reshape(n)[tuple(row_indices.T)]

    # Take svd of rows to extract dominant features
    rows_norm = np.linalg.norm(Rows, axis=1)
    if rescacle_rows:
        Dr, Dr_inv = np.diag(rows_norm), np.diag(1.0 / rows_norm)
        u, s, vh = scila.svd(Dr_inv @ Rows, full_matrices=False)
        u = Dr @ u
    else:
        u, s, vh = scila.svd(Rows, full_matrices=False)
    u, s, vh = u[:, :rank], s[:rank], vh[:rank, :]

    # Initalize X0 with RBFs.  # TODO: better method for X0.
    sigma = 0.75 * np.linalg.norm(row_indices[1] - row_indices[0])
    weights = create_interpolation_weights(n, row_indices, sigma=sigma)
    X0 = (weights * window).reshape(-1, np.prod(n)).T
    # Conbine with SVD of Rows
    X0 = X0 @ (u @ np.diag(s)).real
    Qh = vh

    # Solve XR problem
    C = Cols
    # alpha = len(ic) / np.prod(n) # too small
    alpha = (np.linalg.norm(Qh[:, ic], "fro") / np.linalg.norm(Qh, "fro")) ** 2
    print(f"{alpha = }")
    A1 = Qh[:, ic]
    A2 = Qh
    B1 = C - X0 @ Qh[:, ic]
    dX = (
        B1 @ np.conj(A1.T) @ scila.inv(A1 @ np.conj(A1.T) + alpha * A2 @ np.conj(A2.T))
    ).real
    X = X0 + dX

    pdo_c = X.T.reshape(rank, *n)
    pdo_r = Qh.reshape(rank, *n)
    return pdo_c, pdo_r


def save_pdo(path: str, pdo_c: np.ndarray, pdo_r: np.ndarray, max_plots: int = 10):
    path: Path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "pdo_c.npy", pdo_c)
    np.save(path / "pdo_r.npy", pdo_r)
    plot_psido(pdo_c, pdo_r, max_rows=max_plots)
    plt.savefig(path / "pdo.png")


def save_symbol_cols(
    path: str,
    cols: np.ndarray,
    col_indices: np.ndarray,
    info: dict[str, np.ndarray],
    show_proc: bool = False,
):
    path: Path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "cols.npy", cols)
    np.save(path / "col_indices.npy", col_indices)
    for idx_action, action_info in enumerate(info):
        cols_i = action_info["cols"]
        col_indices_i = action_info["indices"]
        v = action_info["v"]
        Nv = action_info["op_v"]
        np.save(path / f"cols_{idx_action}.npy", cols_i)
        np.save(path / f"col_indices_{idx_action}.npy", col_indices_i)
        np.save(path / f"col_v_{idx_action}.npy", v)
        np.save(path / f"col_Nv_{idx_action}.npy", Nv)
        if not np.iscomplexobj(v):
            fig_v, _ = plot_vec_with_fft(v, f"$v_{idx_action}$", "Specturm")
            plt.savefig(path / f"col_v_{idx_action}.png")
            fig_Nv, _ = plot_vec_with_fft(Nv, f"$Hv_{idx_action}$", "Specturm")
            plt.savefig(path / f"col_Nv_{idx_action}.png")
        fig_proc, _ = plot_pdo_process(cols_i, col_indices_i, Nv)
        plt.savefig(path / f"cols_{idx_action}.png")
        if not show_proc:
            plt.close(fig_proc)


def save_symbol_rows(
    path: Path,
    rows: np.ndarray,
    row_indices: np.ndarray,
    info: dict[str, np.ndarray],
    show_proc: bool = False,
):
    path: Path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "rows.npy", rows)
    np.save(path / "row_indices.npy", row_indices)
    for idx_action, action_info in enumerate(info):
        rows_i = action_info["rows"]
        row_indices_i = action_info["indices"]
        v = action_info["v"]
        Nv = action_info["op_v"]
        np.save(path / f"rows_{idx_action}.npy", rows_i)
        np.save(path / f"row_indices_{idx_action}.npy", row_indices_i)
        np.save(path / f"row_v_{idx_action}.npy", v)
        np.save(path / f"row_Nv_{idx_action}.npy", Nv)
        fig_v, _ = plot_vec(v, f"$v_{idx_action}$")
        plt.savefig(path / f"row_v_{idx_action}.png")
        fig_Nv, _ = plot_vec(Nv, f"$Hv_{idx_action}$")
        plt.savefig(path / f"row_Nv_{idx_action}.png")
        fig_proc, _ = plot_psf_process(rows_i, row_indices_i, Nv)
        plt.savefig(path / f"rows_{idx_action}.png")
        if not show_proc:
            plt.close(fig_proc)
