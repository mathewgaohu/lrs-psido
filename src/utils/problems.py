from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
from scipy import fft, ndimage, sparse

from lrs_psido.linear_operators import RealPDOLowRankSymbol
from utils.bilaplacian_prior import (
    BiLaplacianComputeCoefficients,
    BilaplacianPrior,
    BilaplacianPriorWithoutBC,
)
from utils.make_window import make_window, tukey_window_2d
from utils.wave_modeling_julia import WaveModeling, WaveModelingFG


@dataclass
class SeismicInverseProblem:

    # Geometry
    Lx: float
    Ly: float
    nx: int
    ny: int

    # Model
    cmin: float
    cmax: float
    smooth_sigma: float = None  # Smooth factor for initial guess

    # Frequency
    fmin: float = np.nan
    fmax: float = np.nan

    # Experiment settings
    n_src: int = np.nan
    n_rcv: int = np.nan
    T: float = np.nan
    nt: int = np.nan

    # Dictionary config
    window_config: dict = None
    prior_config: dict = None
    misfit_config: dict = None
    hessian_config: dict = None  # Config Hessian for toy model
    plot_config: dict = None

    label: str = None  # Experiment label
    path: Path = None  # Experiment path
    fast_cost: bool = False  # Whether skip g when evaluate f

    def __post_init__(self):
        self.path = Path(self.path)
        self.get_geometry()
        self.get_model()
        self.get_window()
        self.get_prior()
        self.get_misfit()
        self.get_plotters()
        self.is_toy = self.hessian_config is not None

    def get_geometry(self):
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.L = np.array([self.Lx, self.Ly]).astype(float)
        self.n = np.array([self.nx, self.ny]).astype(int)
        self.d = np.array([self.dx, self.dy]).astype(float)
        self.N = np.prod(self.n)
        self.kmax = np.pi / self.d  # max wavenumber
        self.dt: float = self.T / (self.nt - 1)

        # Grids (may be used at some time)
        xx = np.linspace(0, self.Lx, self.nx)
        yy = np.linspace(0, self.Ly, self.ny)
        xxf = fft.fftfreq(self.nx, self.dx) * 2 * np.pi * self.nx / (self.nx - 1)
        yyf = fft.fftfreq(self.ny, self.dy) * 2 * np.pi * self.ny / (self.ny - 1)
        self.XX, self.YY = np.meshgrid(xx, yy, indexing="ij")
        self.XXf, self.YYf = np.meshgrid(xxf, yyf, indexing="ij")
        self.RRf = np.sqrt(self.XXf**2 + self.YYf**2)
        self.AAf = np.arctan2(self.YYf, self.XXf)

    def get_model(self):
        self.mt: np.ndarray = np.load(self.path / "mt.npy").astype(float).reshape(-1)
        np.clip(self.mt, self.cmin, self.cmax, out=self.mt)

        # Initial Model
        m0_path = self.path / "m0.npy"
        if m0_path.is_file():
            self.m0: np.ndarray = np.load(m0_path).astype(float).reshape(-1)
        else:
            if self.smooth_sigma is not None:
                self.m0 = ndimage.gaussian_filter(
                    self.mt.reshape(self.n), sigma=self.smooth_sigma, mode="nearest"
                ).reshape(-1)
            else:
                self.m0 = np.zeros_like(self.mt)

    def get_window(self):
        method = self.window_config.get("method", "exp")
        if method == "exp":
            window = make_window(
                self.n,
                boundary=(
                    self.window_config["left"],
                    self.window_config["right"],
                    self.window_config["up"],
                    self.window_config["down"],
                ),
                ex=self.window_config["ex"],
                ll=self.window_config["ll"],
            )
        elif method == "tukey":
            alpha_x = self.window_config["alpha_x"]
            alpha_y = self.window_config["alpha_y"]
            window = tukey_window_2d(self.n, alpha=(alpha_x, alpha_y))
        else:
            raise ValueError(f"Unknown method: {method}")

        if self.hessian_config is None:
            # Zero out boundaries where the seismic solver is not correct.
            window[0, :] = 0.0
            window[-1, :] = 0.0
            window[:, 0] = 0.0
            window[:, -1] = 0.0
        self.window = window
        self.W = spla.aslinearoperator(sparse.diags(window.ravel()))

    def get_prior(self):
        name = self.prior_config.get("name")
        if name == "bilaplacian":
            default_rho = self.m0.min() / self.fmax * 0.5
            rho = self.prior_config.get("rho", default_rho)  # correlation length
            sigma2 = self.prior_config.get("sigma2")  # marginal variance
            gamma, delta = BiLaplacianComputeCoefficients(sigma2, rho, ndim=2)
            print(f"Prior: {delta = :.4e}, {gamma = :.4e}")
            self.prior = BilaplacianPrior(
                *self.n,
                *self.d,
                gamma * np.sqrt(self.d.prod()),
                delta * np.sqrt(self.d.prod()),
                mean=np.zeros(self.N),
            )
        elif name == "bilaplacian-without-bc":
            default_rho = self.m0.min() / self.fmax * 0.5
            rho = self.prior_config.get("rho", default_rho)  # correlation length
            sigma2 = self.prior_config.get("sigma2")  # marginal variance
            gamma, delta = BiLaplacianComputeCoefficients(sigma2, rho, ndim=2)
            print(f"Prior: {delta = :.4e}, {gamma = :.4e}")
            self.prior = BilaplacianPriorWithoutBC(
                *self.n,
                *self.d,
                gamma * np.sqrt(self.d.prod()),
                delta * np.sqrt(self.d.prod()),
                mean=np.zeros(self.N),
            )
        else:
            raise ValueError(f"Unknown prior name {name}")

        self.R = spla.aslinearoperator(self.prior.R)
        self.sqrtR = spla.aslinearoperator(self.prior.Rh)
        solver_tol = np.finfo(np.float64).eps
        sqrtRsolve = lambda x: spla.cg(self.prior.Rh, x, rtol=solver_tol)[0]
        self.sqrtRinv = spla.LinearOperator(
            dtype=np.float64,
            shape=self.sqrtR.shape,
            matvec=sqrtRsolve,
            rmatvec=sqrtRsolve,
        )
        self.Rinv = self.sqrtRinv @ self.sqrtRinv.H

    def get_misfit(self):
        if self.hessian_config:
            self.get_toy_misfit()  # known hessian, linear model
        else:
            self.get_seismic_misfit()

    def get_seismic_misfit(self):
        if self.fast_cost:
            self.client = WaveModeling("/tmp/mathew-seismic", timeout=120)
        else:
            self.client = WaveModelingFG("/tmp/mathew-seismic", timeout=120)
        noise_var = self.misfit_config["noise_variance"]

        # Misfit and derivatives
        def orig_misfit_cost(x: np.ndarray) -> float:
            return self.client.compute_f(self.m0 + x) / noise_var

        def orig_misfit_grad(x: np.ndarray) -> np.ndarray:
            return self.client.compute_g(self.m0 + x) / noise_var

        def orig_misfit_hess(x: np.ndarray) -> spla.LinearOperator:
            return spla.LinearOperator(
                shape=(self.N, self.N),
                dtype=float,
                matvec=lambda v: self.client.compute_H(self.m0 + x, v) / noise_var,
                rmatvec=lambda v: self.client.compute_H(self.m0 + x, v) / noise_var,
            )

        def misfit_cost(x: np.ndarray) -> float:
            return orig_misfit_cost(self.W * x)

        def misfit_grad(x: np.ndarray) -> np.ndarray:
            return self.W * orig_misfit_grad(self.W * x)

        def misfit_hess(x: np.ndarray) -> spla.LinearOperator:
            return self.W * orig_misfit_hess(self.W * x) * self.W

        self._misfit_cost = misfit_cost
        self._misfit_grad = misfit_grad
        self._misfit_hess = misfit_hess
        self.orig_misfit_cost = orig_misfit_cost
        self.orig_misfit_grad = orig_misfit_grad
        self.orig_misfit_hess = orig_misfit_hess

    def get_toy_misfit(self) -> None:
        lr_rank: int = self.hessian_config["lr_rank"]
        lr_weight: float = self.hessian_config["lr_weight"]
        pdo_weight: float = self.hessian_config.get("pdo_weight", 1.0)
        depth_factor: str = self.hessian_config["depth_factor"]

        XX0: np.ndarray = self.XX / self.XX.max()
        YY0: np.ndarray = self.YY / self.YY.max()

        if depth_factor == "d2":
            D = (YY0**2 + 0.1) ** (-1)
        elif depth_factor == "m2":
            D = (self.mt.reshape(self.n) / self.cmin) ** (-2) * (YY0 * 2 + 1) ** (-2)
        else:
            raise ValueError(f"Unknown {depth_factor = }")

        D /= np.max(D)

        # Build Low-rank part
        evals = (1.0 + np.arange(lr_rank)) ** (-0.5) * lr_weight
        evecs = np.stack([np.sin(k * np.pi * XX0) * D for k in range(1, lr_rank + 1)])
        evecs = evecs.reshape(lr_rank, self.N).T
        evecs /= np.linalg.norm(evecs, axis=0)
        E = spla.aslinearoperator(sparse.diags(evals))
        V = spla.aslinearoperator(evecs)
        H_lr = V @ E @ V.H

        # Build POD rows (order=0.5; trim high frequencies)
        low_pass = lambda r, a, b: (r <= a) + (r > a) * (r < b) * (b - r) / (b - a)
        rows = (
            np.stack(
                [
                    0.5 * np.ones_like(self.AAf) + 0.1,
                    0.5 * np.cos(2 * self.AAf),
                    -0.5 * np.sin(2 * self.AAf),
                ]
            )
            * self.RRf**0.5
            * low_pass(self.RRf, 0.5 * self.kmax[0], 0.7 * self.kmax[0])
        )

        # Build PDO columns (windowed; decreasing-on-depth)
        cols = (
            self.window
            * D
            * np.stack(
                [
                    np.ones_like(XX0),
                    np.cos(2 * np.pi * XX0),
                    np.sin(2 * np.pi * XX0),
                ]
            )
        ) * pdo_weight

        # Hessian=Hlr+Hpdo
        H_pdo = RealPDOLowRankSymbol(cols, rows)
        H_true = H_lr + H_pdo * H_pdo.H

        # coefficients of the misfit
        b = H_true * (self.mt - self.m0)
        c = 0.5 * np.dot(b, self.mt - self.m0)

        # Misfit and derivatives
        def orig_misfit_cost(x: np.ndarray) -> float:
            return 0.5 * np.dot(H_true * x, x) - np.dot(b, x) + c

        def orig_misfit_grad(x: np.ndarray) -> np.ndarray:
            return H_true * x - b

        def orig_misfit_hess(x: np.ndarray) -> spla.LinearOperator:
            return H_true

        def misfit_cost(x: np.ndarray) -> float:
            return orig_misfit_cost(self.W * x)

        def misfit_grad(x: np.ndarray) -> np.ndarray:
            return self.W * orig_misfit_grad(self.W * x)

        def misfit_hess(x: np.ndarray) -> spla.LinearOperator:
            return self.W * orig_misfit_hess(self.W * x) * self.W

        self._misfit_cost = misfit_cost
        self._misfit_grad = misfit_grad
        self._misfit_hess = misfit_hess
        self.orig_misfit_cost = orig_misfit_cost
        self.orig_misfit_grad = orig_misfit_grad
        self.orig_misfit_hess = orig_misfit_hess

        self.H_true = H_true
        self.b = b
        self.evals = evals
        self.evecs = evecs
        self.rows = rows
        self.cols = cols

    def misfit_cost(self, x: np.ndarray) -> float:
        return self._misfit_cost(x)

    def misfit_grad(self, x: np.ndarray) -> np.ndarray:
        return self._misfit_grad(x)

    def misfit_hess(self, x: np.ndarray) -> spla.LinearOperator:
        return self._misfit_hess(x)

    def total_cost(self, x: np.ndarray) -> float:
        """cost(x) = misfit(m0+W*x) + reg(x)"""
        misfit = self.misfit_cost(x)
        reg = self.prior.cost(x)
        total = misfit + reg
        # if not self.fast_cost:
        #     print(f"{misfit = :.6e}, {reg = :.6e}")
        return total

    def total_grad(self, x: np.ndarray) -> np.ndarray:
        return self.misfit_grad(x) + self.prior.grad(x)

    def total_hess(self, x: np.ndarray) -> spla.LinearOperator:
        return self.misfit_hess(x) + self.R

    def get_plotters(self) -> None:
        plt.rcParams.update(self.plot_config["rcParams"])
        self.fig_width = self.plot_config["width"]
        self.x_ratio = self.plot_config["x_ratio"]
        self.f_ratio = self.plot_config["f_ratio"]
        self.x_size = np.asarray((self.fig_width / 2, self.fig_width / 2 / self.x_ratio))
        self.f_size = np.asarray((5, 5 / self.f_ratio))

    def plot(self, x: np.ndarray, title: str = None, colorbar: bool = False, **kwargs):
        kwargs.setdefault("extent", [0, self.Lx, self.Ly, 0])
        kwargs.setdefault("aspect", 1)
        im = plt.imshow(x.reshape(self.n).T, **kwargs)
        if title is not None:
            plt.title(title)
        if colorbar:
            plt.colorbar()
        return im

    def plotf(self, x: np.ndarray, title: str = None, colorbar: bool = False, **kwargs):
        kwargs.setdefault(
            "extent", [-self.kmax[0], self.kmax[0], self.kmax[1], -self.kmax[1]]
        )
        kwargs.setdefault("aspect", 1)
        im = plt.imshow(fft.fftshift(x.reshape(self.n).T), **kwargs)
        if title is not None:
            plt.title(title)
        if colorbar:
            plt.colorbar()
        return im

    def plot_pictures(self, n_evals: int = 300) -> None:
        out_path = self.path / "model"
        out_path.mkdir(parents=True, exist_ok=True)

        # plot window
        plt.figure(figsize=self.x_size)
        self.plot(self.window)
        plt.colorbar()
        plt.title("Window")
        plt.tight_layout()
        plt.savefig(out_path / "window.png")

        # plot mture
        plt.figure(figsize=self.x_size)
        self.plot(self.mt)
        plt.colorbar()
        plt.title("True Model")
        plt.tight_layout()
        plt.savefig(out_path / "mt.png")
        np.save(out_path / "mt.npy", self.mt.reshape(self.n))

        # plot m0
        plt.figure(figsize=self.x_size)
        self.plot(self.m0)
        plt.colorbar()
        plt.title("Initial Model")
        plt.tight_layout()
        plt.savefig(out_path / "m0.png")
        np.save(out_path / "m0.npy", self.m0.reshape(self.n))

        # plot dmt
        dmt = self.mt - self.m0
        plt.figure(figsize=self.x_size)
        self.plot(dmt)
        plt.colorbar()
        plt.title(r"Target solution $x^* = m_{true}-m_0$")
        plt.tight_layout()
        plt.savefig(out_path / "dmt.png")

        if self.hessian_config:
            plt.figure(figsize=self.x_size)
            self.plot(self.b)
            plt.colorbar()
            plt.title(r"$b=Hx^*$")
            plt.tight_layout()
            plt.savefig(out_path / "b.png")

            evals_H, evecs_H = spla.eigsh(
                self.H_true, k=n_evals, return_eigenvectors=True
            )
            evals_H, evecs_H = evals_H[::-1], evecs_H[:, ::-1]
            evals_R = np.linalg.norm(self.sqrtR @ evecs_H, axis=0) ** 2
            fig = plt.figure(figsize=(6, 4))
            plt.semilogy(evals_H, label="$H$")
            plt.semilogy(self.evals, label=r"$H_{lr}$")
            plt.semilogy(evals_R, label="$R$")
            plt.title("Eigenvalues of the operator")
            plt.legend()
            plt.savefig(out_path / "evals.png")

            k = len(self.evals)
            fig, axes = plt.subplots(k, 2, figsize=(6 * 2, 3 * k))
            for i in range(k):
                plt.sca(axes[i, 0])
                plt.imshow(self.evecs[:, i].reshape(self.n).T)
                plt.colorbar()
                plt.title(f"No.{i+1} eigenvector of " + r"$H_{lr}$")
                plt.sca(axes[i, 1])
                plt.imshow(evecs_H[:, i].reshape(self.n).T)
                plt.colorbar()
                plt.title(f"No.{i+1} eigenvector of " + r"$H$")
            plt.tight_layout()
            plt.savefig(out_path / "evecs.png")

            HR = self.total_hess(self.m0)
            g0 = self.total_grad(self.m0)
            dm_solve = spla.cg(HR, -g0)[0]
            dm_solve = self.W * dm_solve
            plt.figure(figsize=self.x_size)
            self.plot(dm_solve)
            plt.colorbar()
            plt.title(r"Solution $x = (H+R)^{-1} b$")
            plt.tight_layout()
            plt.savefig(out_path / "dm_solve.png")

            k = len(self.cols)
            width = 14
            width_ratios = (self.x_ratio, self.x_ratio, self.f_ratio)
            fig, axes = plt.subplots(
                k,
                3,
                figsize=(width, width / sum(width_ratios) * k),
                width_ratios=width_ratios,
            )
            for i in range(k):
                plt.sca(axes[i, 0])
                self.plot(self.evecs[:, i], title=f"$v_{i+1}(x)$", colorbar=True)
                plt.sca(axes[i, 1])
                self.plot(self.cols[i], title=f"$a_{i+1}(x)$", colorbar=True)
                plt.sca(axes[i, 2])
                self.plotf(self.rows[i], title=f"$b_{i+1}" + r"(\xi)$", colorbar=True)
            plt.tight_layout()
            plt.savefig(out_path / "lr_pdo.png")


def get_toy() -> SeismicInverseProblem:
    ...

def get_marmousi() -> SeismicInverseProblem:
    ...


def test() -> None:
    sip1 = get_toy()
    sip1.plot_pictures()

    sip2 = get_marmousi()
    sip2.plot_pictures()

if __name__ == "__main__":
    test()
