import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import fft
from skimage.metrics import structural_similarity as ssim

from .problems import SeismicInverseProblem


@dataclass
class OptimResult:
    xx: np.ndarray = field(repr=False)
    ff: np.ndarray = field(repr=False)
    gg: np.ndarray = field(repr=False)
    pp: np.ndarray = field(repr=False)
    label: str = ""
    path: Path = None
    info: dict = None


def get_optim_result(path: Path, label: str = "Computed") -> OptimResult:
    path = Path(path)
    info_path = path / "info.json"
    if info_path.exists():
        with open(info_path, "r") as file:
            info = json.load(file)
    results = OptimResult(
        xx=np.load(path / "xx.npy"),
        ff=np.load(path / "ff.npy"),
        gg=np.load(path / "gg.npy"),
        pp=np.load(path / "pp.npy"),
        label=label,
        path=path,
        info=info,
    )
    return results


@dataclass
class FWIResultsAnalyzer:

    sip: SeismicInverseProblem
    results: Sequence[OptimResult]
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.N = len(self.results)
        self.all_mm: list[np.ndarray] = self.get_all_mm()
        self.all_reg: list[np.ndarray] = self.get_all_reg()
        self.all_mis: list[np.ndarray] = self.get_all_mis()
        self.all_ggn: list[np.ndarray] = self.get_all_ggn()

    def get_all_mm(self) -> list[np.ndarray]:
        all_mm = [self.sip.m0 + (self.sip.W @ res.xx.T).T for res in self.results]
        return all_mm

    def get_all_reg(self) -> list[np.ndarray]:
        all_reg = [
            np.asarray([self.sip.prior.cost(x) for x in res.xx]) for res in self.results
        ]
        return all_reg

    def get_all_mis(self) -> list[np.ndarray]:
        if not hasattr(self, "all_reg"):
            self.all_reg: list[np.ndarray] = self.get_all_reg()

        all_mis = [res.ff - reg for res, reg in zip(self.results, self.all_reg)]
        return all_mis

    def get_all_ggn(self) -> list[np.ndarray]:
        ggn = [np.linalg.norm(res.gg, axis=1) for res in self.results]
        return ggn

    def plot_all(
        self,
        maxiter: int = 30,
        maxiter_image: int = 20,
        show_iter: int = 10,
        offset: int = None,
    ) -> None:
        self.plot_mis(maxiter)
        self.plot_ggn()
        self.plot_err(maxiter)
        # self.plot_ssim(maxiter)
        self.plot_model_vs_depth(it=show_iter, offset=offset)
        self.plot_model_err_vs_depth(it=show_iter, offset=offset)
        if not hasattr(self.sip, "hessian_config"):
            self.plot_data_fft(it=show_iter)

        if self.N == 1:  # Single result analysis
            self.plot_ff(maxiter)
            # self.plot_ssim_img(maxiter_image)
            self.plot_updates(maxiter_image)
            if not hasattr(self.sip, "hessian_config"):
                self.plot_data(src_idx=self.sip.n_src // 2)

        if self.N > 1:
            self.compare_models(it=show_iter)

    def plot_ff(self, maxiter: int) -> None:
        """Plot misfit and regularization"""
        assert self.N == 1
        mis, reg = self.all_mis[0], self.all_reg[0]
        fig = plt.figure()
        plt.title("Misfit and Regularization")
        plt.plot(mis[: maxiter + 1], "-o", label="Misfit")
        plt.plot(reg[: maxiter + 1], "-o", label="Regularization")
        plt.xlabel("Iterations")
        plt.legend()
        plt.savefig(self.output_dir / "ff.png")

    def plot_mis(self, maxiter: int) -> None:
        fig = plt.figure()
        for res, mis in zip(self.results, self.all_mis):
            plt.semilogy(mis[: maxiter + 1], "-o", label=res.label)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Misfit")
        plt.xlabel("Iterations")
        plt.ylabel(r"$\frac{1}{2} \| f(m)-d\|^2_{\Gamma_{noise}^{-1}}$")
        plt.legend()
        plt.savefig(self.output_dir / "mis.png")

    def plot_ggn(self) -> None:
        """Plot gradient"""
        fig = plt.figure()
        for res, ggn in zip(self.results, self.all_ggn):
            plt.semilogy(ggn / ggn[0], label=res.label)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Gradient Norm")
        plt.xlabel("Iterations")
        plt.ylabel(r"$|g|/|g_0|$")
        plt.legend()
        plt.savefig(self.output_dir / "ggn.png")

    def plot_err(self, maxiter: int) -> None:
        """Solution L2 Error"""
        mt = self.sip.mt
        fig = plt.figure()
        for res, mm in zip(self.results, self.all_mm):
            ee = np.linalg.norm(mm - mt, axis=1) / np.linalg.norm(mt)
            plt.plot(ee[: maxiter + 1], "-o", label=res.label)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Solution Error")
        plt.xlabel("Iterations")
        plt.ylabel(r"$|m-m_{true}|_2/|m_{true}|_2$")
        plt.legend()
        plt.savefig(self.output_dir / "err_l2.png")

    def plot_ssim(self, maxiter: int):
        """SSIM Error"""
        n = self.sip.n
        mt = self.sip.mt
        data_range = mt.max() - mt.min()
        fig = plt.figure()
        for res, mm in zip(self.results, self.all_mm):
            ee = np.array(
                [ssim(m.reshape(n), mt.reshape(n), data_range=data_range) for m in mm]
            )
            plt.plot(ee[: maxiter + 1], "-o", label=res.label)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("SSIM")
        plt.xlabel("Iterations")
        plt.ylabel(r"$SSIM(m-m_{true})$")
        plt.legend()
        plt.savefig(self.output_dir / "ssim.png")

    def plot_ssim_img(self, maxiter: int):
        assert self.N == 1
        mm = self.all_mm[0]
        n = self.sip.n
        mt = self.sip.mt
        data_range = mt.max() - mt.min()

        fig, axs = plt.subplots(
            maxiter, 1, figsize=self.sip.x_size * np.array([1, maxiter])
        )
        for i in range(maxiter):
            ssim_val, ssim_img = ssim(
                mm[i].reshape(n), mt.reshape(n), data_range=data_range, full=True
            )
            plt.sca(axs[i])
            self.sip.plot(ssim_img)
            plt.colorbar()
            plt.title(f"Iter {i}: SSIM={ssim_val}")
        plt.tight_layout()
        plt.savefig(self.output_dir / "ssim_img.png")

    def plot_updates(self, maxiter: int):
        """Plot updates (x, g, p)"""
        assert self.N == 1
        xx = self.results[0].xx
        gg = self.results[0].gg
        pp = self.results[0].pp
        mm = self.all_mm[0]
        mis = self.all_mis[0]
        reg = self.all_reg[0]
        ggn = self.all_ggn[0]
        fig, axs = plt.subplots(
            maxiter, 3, figsize=self.sip.x_size * np.array([3, maxiter])
        )
        for i in range(maxiter):
            plt.sca(axs[i, 0])
            self.sip.plot(mm[i])
            plt.colorbar()
            plt.title(f"Iter {i+1} - m: misft={mis[i]:.2e}, reg={reg[i]:.2e}")
            plt.sca(axs[i, 1])
            self.sip.plot(gg[i])
            plt.colorbar()
            plt.title(f"Iter {i+1} - g: $|g|$={ggn[i]:.2e}")
            plt.sca(axs[i, 2])
            self.sip.plot(pp[i])
            plt.colorbar()
            step_size = np.linalg.norm(xx[i + 1] - xx[i]) / np.linalg.norm(pp[i])
            plt.title(f"Iter {i+1} - p: step size={step_size:.2e}")
        plt.tight_layout()
        plt.savefig(self.output_dir / "udpates.png")

    def plot_model_vs_depth(self, it: int, offset: int = None) -> None:
        """Plot model vs depth.

        Args:
            it: Number of iteration
            i: Index in x direction.
        """
        n = self.sip.n
        mt = self.sip.mt.reshape(n)
        if offset is None:
            offset = mt.shape[0] // 2
        dd = np.arange(mt.shape[1]) * self.sip.dy
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        plt.plot(mt[offset, :], dd, "k", label="True Model")
        for res, mm in zip(self.results, self.all_mm):
            m = mm[it].reshape(n)
            plt.plot(m[offset, :], dd, label=res.label)
        plt.title(f"Models at Offset={offset * self.sip.dx:.0f} m")
        plt.gca().invert_yaxis()
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Depth (m)")
        plt.legend()
        plt.savefig(self.output_dir / "model_vs_depth.png")

    def plot_model_err_vs_depth(self, it: int, offset: int = None) -> None:
        """Plot model error vs depth"""
        pass

    def plot_data(self, src_idx: int = 40, iters: Sequence[int] = None) -> None:
        pass

    def get_data(self, m: np.ndarray, path: Path) -> np.ndarray:
        if path.is_file():
            return np.load(path)
        else:
            data = self.sip.client.compute_d(m)
            np.save(path, data)
            return data

    def plot_data_fft(self, it: int, max_Hz: float = 15.0) -> None:
        """Plot Data Error(s,r,t)'s FFT"""
        mt = self.sip.mt
        data_true_path = self.results[0].path / "data_true.npy"
        data_true = self.get_data(mt, data_true_path)
        hzs = fft.fftfreq(data_true.shape[-1], self.sip.dt)
        imax = int(np.round(max_Hz * self.sip.T))
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        for res, mm in zip(self.results, self.all_mm):
            data = self.get_data(mm[it], res.path / f"data_{it}.npy")
            d_err = data - data_true
            ee = np.sum(np.abs(fft.fft(d_err, axis=-1)), axis=(0, 1))
            plt.plot(hzs[:imax], ee[:imax], label=res.label)
        plt.title("Error in Different Freqeuncies")
        plt.xlabel("Hz")
        plt.legend()
        plt.savefig(self.output_dir / "data_error_fft.png")

    def compare_models(self, it: int, ncols: int = 3, same_range=True) -> None:
        nrows = np.ceil((self.N + 1) / ncols).astype(int)
        xt = self.sip.mt - self.sip.m0
        kwargs = {}
        if same_range:
            kwargs["vmin"] = xt.min()
            kwargs["vmax"] = xt.max()
        fig, axs = plt.subplots(
            nrows, ncols, figsize=self.sip.x_size * np.array([ncols, nrows])
        )
        plt.sca(axs.flat[0])
        self.sip.plot(
            xt,
            title=r"Target Solution $(m_{true}-m_0)$",
            colorbar=True,
            cbar_label="Velocity (m/s)",
            **kwargs,
        )
        for i, res in enumerate(self.results):
            plt.sca(axs.flat[i + 1])
            self.sip.plot(
                self.sip.W * res.xx[it],
                title=f"{res.label} - {it}th Iteration ($m_k-m_0$)",
                colorbar=True,
                cbar_label="Velocity (m/s)",
                **kwargs,
            )
        plt.tight_layout()
        plt.savefig(self.output_dir / f"compare_x{it}.png")


class NewtonCGResultsAnalyzer(FWIResultsAnalyzer):

    @cached_property
    def hess_counts(self) -> list[np.ndarray]:
        output = []
        for res in self.results:
            hess_count_history: list = res.info["hess_count_history"]
            hess_counts = np.cumsum([0] + hess_count_history)
            output.append(hess_counts)
        return output

    def plot_mis(self, maxiter: int, max_hess_counts: int = 1000) -> None:
        fig = plt.figure()
        for res, mis, hess_counts in zip(self.results, self.all_mis, self.hess_counts):
            maxiter = min(maxiter, sum(hess_counts < max_hess_counts) - 1)
            plt.semilogy(
                hess_counts[: maxiter + 1], mis[: maxiter + 1], "-o", label=res.label
            )
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Misfit")
        plt.xlabel("Number of Hessian Actions")
        plt.ylabel(r"$\frac{1}{2} \| f(m)-d\|^2_{\Gamma_{noise}^{-1}}$")
        plt.legend()
        plt.savefig(self.output_dir / "mis.png")

    def plot_ggn(self, maxiter: int = 1000, max_hess_counts: int = 1000) -> None:
        """Plot gradient"""
        fig = plt.figure()
        for res, ggn, hess_counts in zip(self.results, self.all_ggn, self.hess_counts):
            maxiter = min(maxiter, sum(hess_counts < max_hess_counts) - 1)
            plt.semilogy(
                hess_counts[: maxiter + 1], ggn[: maxiter + 1] / ggn[0], label=res.label
            )
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Gradient Norm")
        plt.xlabel("Number of Hessian Actions")
        plt.ylabel(r"$|g|/|g_0|$")
        plt.legend()
        plt.savefig(self.output_dir / "ggn.png")

    def plot_err(self, maxiter: int, max_hess_counts: int = 1000) -> None:
        """Solution L2 Error"""
        mt = self.sip.mt
        fig = plt.figure()
        for res, mm, hess_counts in zip(self.results, self.all_mm, self.hess_counts):
            maxiter = min(maxiter, sum(hess_counts < max_hess_counts) - 1)
            ee = np.linalg.norm(mm - mt, axis=1) / np.linalg.norm(mt)
            plt.plot(
                hess_counts[: maxiter + 1], ee[: maxiter + 1], "-o", label=res.label
            )
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Solution Error")
        plt.xlabel("Number of Hessian Actions")
        plt.ylabel(r"$|m-m_{true}|_2/|m_{true}|_2$")
        plt.legend()
        plt.savefig(self.output_dir / "err_l2.png")
