from typing import Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator


class GaussianLowPassFilter(LinearOperator):
    """Low-pass filter implemented with Gaussian kernel."""

    def __init__(self, image_shape: Sequence[int], sigma: float):
        self.image_shape = image_shape
        self.sigma = sigma
        super().__init__(
            dtype=float,
            shape=(np.prod(image_shape), np.prod(image_shape)),
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(self.image_shape)
        blurred = ndimage.gaussian_filter(x, sigma=self.sigma, mode="nearest")
        return blurred.reshape(-1)

    def _adjoint(self):
        return GaussianLowPassFilter(self.image_shape, self.sigma)


class GaussianHighPassFilter(LinearOperator):
    """High-pass filter implemented with Gaussian kernel."""

    def __init__(self, image_shape: Sequence[int], sigma: float):
        self.image_shape = image_shape
        self.sigma = sigma
        super().__init__(
            dtype=float,
            shape=(np.prod(image_shape), np.prod(image_shape)),
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(self.image_shape)
        blurred = ndimage.gaussian_filter(x, sigma=self.sigma, mode="nearest")
        high_pass = x - blurred
        return high_pass.reshape(-1)

    def _adjoint(self):
        return GaussianHighPassFilter(self.image_shape, self.sigma)
