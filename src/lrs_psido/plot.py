import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import fft


def plot_psido(
    pdo_c: np.ndarray,
    pdo_r: np.ndarray,
    real_only: bool = False,
    max_rows: int = 10,
) -> tuple[Figure, Axes]:
    nrows = min(len(pdo_c), max_rows)
    n = np.asarray(pdo_c.shape[1:])
    hight = 3
    x_ratio = n[0] / n[1]
    f_ratio = 1

    if real_only:
        pdo_c = pdo_c.real
        pdo_r = pdo_r.real
        width_ratios = (x_ratio, f_ratio)
        figsize = (hight * sum(width_ratios), hight * nrows)
    else:
        width_ratios = (x_ratio, f_ratio, f_ratio)
        figsize = (hight * sum(width_ratios), hight * nrows)

    fig, axs = plt.subplots(
        nrows, len(width_ratios), figsize=figsize, width_ratios=width_ratios
    )
    for i in range(nrows):
        plt.sca(axs[i, 0])
        plt.imshow(pdo_c[i].T, aspect=1)
        plt.colorbar()
        plt.title(f"$a_{{{i+1}}}(x)$")
        plt.sca(axs[i, 1])
        if real_only:
            plt.imshow(fft.fftshift(pdo_r[i].T), aspect=n[0] / n[1])
            plt.colorbar()
            plt.title(f"$b_{{{i+1}}}(\\xi)$")
        else:
            plt.imshow(fft.fftshift(pdo_r[i].real.T), aspect=n[0] / n[1])
            plt.colorbar()
            plt.title(f"$b_{{{i+1}}}(\\xi)$ (real)")
            plt.sca(axs[i, 2])
            plt.imshow(fft.fftshift(pdo_r[i].imag.T), aspect=n[0] / n[1])
            plt.colorbar()
            plt.title(f"$b_{{{i+1}}}(\\xi)$ (imaginary)")
    plt.tight_layout()
    return fig, axs


def plot_vec(x: np.ndarray, title: str = "") -> tuple[Figure, Axes]:
    n = np.asarray(x.shape)
    fig_width = 15
    x_ratio = n[0] / n[1]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(fig_width, fig_width / x_ratio),
    )
    plt.imshow(x.T, aspect=1)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    return fig, ax


def plot_vec_with_fft(
    x: np.ndarray, xtitle: str = "", ftitle: str = ""
) -> tuple[Figure, Axes]:
    n = np.asarray(x.shape)
    fig_width = 15
    x_ratio = n[0] / n[1]
    f_ratio = 1

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_width / (x_ratio + f_ratio)),
        width_ratios=(x_ratio, f_ratio),
    )
    plt.sca(axs[0])
    plt.imshow(x.T, aspect=1)
    plt.colorbar()
    plt.title(xtitle)
    plt.sca(axs[1])
    plt.imshow(fft.fftshift(np.abs(fft.fft2(x)).T), aspect=n[0] / n[1])
    plt.colorbar()
    plt.title(ftitle)
    plt.tight_layout()
    return fig, axs


def plot_pdo_process(
    cols: np.ndarray, indices: np.ndarray, Nv: np.ndarray
) -> tuple[Figure, Axes]:
    nrows = len(cols)
    n = np.asarray(cols.shape[1:])
    fig_width = 15
    x_ratio = n[0] / n[1]
    f_ratio = 1

    fig, axs = plt.subplots(
        nrows,
        3,
        figsize=(fig_width, fig_width / (x_ratio * 2 + f_ratio) * nrows),
        width_ratios=(x_ratio, x_ratio, f_ratio),
    )
    axs = np.reshape(axs, (nrows, 3))
    for i in range(nrows):
        col = cols[i]
        index = indices[i]
        plt.sca(axs[i, 0])
        plt.imshow(col.real.T, aspect=1)
        plt.colorbar()
        if i == 0:
            plt.title("symbol column real part")
        plt.sca(axs[i, 1])
        plt.imshow(col.imag.T, aspect=1)
        plt.colorbar()
        if i == 0:
            plt.title("symbol column imaginary part")
        plt.sca(axs[i, 2])
        plt.imshow(fft.fftshift(np.abs(fft.fft2(Nv)).T), aspect=n[0] / n[1])
        plt.plot(
            [fft.ifftshift(np.arange(n[0]))[index[0]]],
            [fft.ifftshift(np.arange(n[1]))[index[1]]],
            "k.",
            markersize=4,
        )
        if i == 0:
            plt.title("sample point")
    plt.tight_layout()
    return fig, axs


def plot_psf_process(
    rows: np.ndarray, indices: np.ndarray, Nv: np.ndarray
) -> tuple[Figure, Axes]:
    nrows = len(rows)
    n = np.asarray(rows.shape[1:])
    fig_width = 15
    x_ratio = n[0] / n[1]
    f_ratio = 1

    fig, axs = plt.subplots(
        nrows,
        3,
        figsize=(fig_width, fig_width / (x_ratio + 2 * f_ratio) * nrows),
        width_ratios=(f_ratio, f_ratio, x_ratio),
    )
    for i in range(nrows):
        row = rows[i]
        index = indices[i]
        plt.sca(axs[i, 0])
        plt.imshow(fft.fftshift(row.real.T), aspect=n[0] / n[1])
        plt.colorbar()
        if i == 0:
            plt.title("symbol row real part")
        plt.sca(axs[i, 1])
        plt.imshow(fft.fftshift(row.imag.T), aspect=n[0] / n[1])
        plt.colorbar()
        if i == 0:
            plt.title("symbol row imaginary part")
        plt.sca(axs[i, 2])
        plt.imshow(Nv.T, aspect=1)
        plt.plot([index[0]], [index[1]], "k.", markersize=4)
        if i == 0:
            plt.title("sample point")
    plt.tight_layout()
    return fig, axs
