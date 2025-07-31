from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mhu_helper_functions.mcmc import (
    integratedAutocorrelationTime,
    plot_autocorrelation,
    plot_hist,
    plot_trace,
)
from scipy import stats

from .problem import SeismicInverseProblem


@dataclass
class MCMCResultsAnalyzer:
    sip: SeismicInverseProblem
    datas: list[np.ndarray]
    labels: list[str]
    colors: list[str]
    qoi_points: Sequence[Sequence[int]]
    output_dir: Path
    true_mean: np.ndarray = field(default=None, repr=False)
    true_std: np.ndarray = field(default=None, repr=False)
    true_color: str = "tab:blue"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.N = len(self.datas)  # Number of comparing experiments
        self.m = len(self.qoi_points)  # Number of sample points for QoI

    def plot_trace_hist_acc(self, selected: Sequence[int], max_lag: int) -> None:
        width = 14  # pdf file's text width
        ax_size = np.array([width / 3, width / 3 * 0.5])
        plot_shape = np.array([self.m, 3])
        fig, axs = plt.subplots(*plot_shape, figsize=ax_size * plot_shape[::-1])
        for i in range(self.m):
            plt.sca(axs[i, 0])
            self.plot_trace(self.qoi_points[i], selected)
            plt.title(r"Trace of Samples at $x_" + f"{i+1}$")
            plt.xlabel("")
            plt.ylabel("")
        self.plot_sorted_legend(selected)
        for i in range(self.m):
            plt.sca(axs[i, 1])
            self.plot_hist(self.qoi_points[i], selected)
            plt.title(r"Histogram of Samples at $x_" + f"{i+1}$")
            plt.xlabel("")
            plt.ylabel("")
            if self.true_std is not None:
                mean = self.true_mean.reshape(self.sip.n)[*self.qoi_points[i]]
                std = self.true_std.reshape(self.sip.n)[*self.qoi_points[i]]
                x = np.linspace(*plt.xlim())
                y = stats.norm.pdf(x, mean, std)
                plt.plot(x, y, label="True distribution", color=self.true_color)
        self.plot_sorted_legend(selected)
        for i in range(self.m):
            plt.sca(axs[i, 2])
            self.plot_acc(self.qoi_points[i], max_lag=max_lag)
            plt.title(r"Auto-correlation of Samples at $x_" + f"{i+1}$")
            plt.xlabel("")
            plt.ylabel("")
        self.plot_sorted_legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "mcmc.png")

    def plot_qoi_points(self, background: np.ndarray) -> None:
        dx, dy = self.sip.dx, self.sip.dy
        plt.figure(figsize=self.sip.x_size)
        self.sip.plot(background)
        plt.colorbar()
        for i, (x, y) in enumerate(self.qoi_points):
            plt.scatter(x * dx, y * dy, color="black")  # Plot the point
            plt.text(x * dx + dx, y * dy, r"$x_" + f"{i+1}$", fontsize=14)
        plt.title("Sample Positions for Quantity of Interest")
        plt.tight_layout()
        plt.savefig(self.output_dir / "qoi.png")

    def plot_trace(self, point: Sequence[int], selected: Sequence[int]) -> None:
        for j in selected:
            q = self.datas[j].reshape(-1, *self.sip.n)[:, *point]
            plot_trace(q, color=self.colors[j], label=self.labels[j])

    def plot_hist(self, point: Sequence[int], selected: Sequence[int]) -> None:
        for j in selected:
            q = self.datas[j].reshape(-1, *self.sip.n)[:, *point]
            plot_hist(q, color=self.colors[j], label=self.labels[j])

    def plot_acc(self, point: Sequence[int], max_lag: int) -> None:
        for j in range(self.N):
            q = self.datas[j].reshape(-1, *self.sip.n)[:, *point]
            plot_autocorrelation(
                q, max_lag=max_lag, color=self.colors[j], label=self.labels[j]
            )

    def plot_sorted_legend(self, selected: Sequence[int] = None) -> None:
        if selected is None:
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1))
            return
        handles, labels = plt.gca().get_legend_handles_labels()
        # First: get additional labels
        if self.true_std is not None:
            new_handles = handles[len(selected) : len(selected) + 1]
            new_labels = labels[len(selected) : len(selected) + 1]
        else:
            new_handles = []
            new_labels = []
        # Second: sort selected labels
        order = np.argsort(selected)
        new_handles = new_handles + [handles[idx] for idx in order]
        new_labels = new_labels + [labels[idx] for idx in order]
        # Plot legend in desired order
        plt.legend(
            new_handles,
            new_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
        )

    def compute_iat(self, max_lag=3000) -> np.ndarray:
        """Compute IAT.

        Args:
            max_log (int): maximum lag, default 3000.

        Returns:
            np.ndarray: A[index_chain, index_sample]
        """
        out = np.zeros((self.N, self.m))
        for i in range(self.N):  # chains
            for j in range(self.m):  # sample points
                q = self.datas[i].reshape(-1, *self.sip.n)[:, *self.qoi_points[j]]
                iat, _, _ = integratedAutocorrelationTime(q, max_lag=max_lag)
                out[i, j] = iat
        return out
