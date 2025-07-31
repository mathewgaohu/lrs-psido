"""Create schemes from given information."""

from typing import Sequence

import numpy as np
from scipy import fft

from .constraints import create_constraint
from .utils import create_equispaced_array, split


class BasicScheme:
    def create_index(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def split(
        self, x: np.ndarray, indices: np.ndarray, roll_to_origin: bool, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError


class EquispaceScheme(BasicScheme):
    def __init__(
        self,
        n: Sequence[int],
        steps: Sequence[int],
        anchor: Sequence[int] = None,
        fft_index: bool = False,
        constraint: dict = {},
        smoothness: float = 0.2,
    ):
        self.n = np.asarray(n)
        self.steps = np.asarray(steps)
        if anchor is not None:
            self.anchor = np.asarray(anchor)
        else:
            self.anchor = np.zeros_like(self.n)
        self.fft_index = fft_index
        self.constraint = create_constraint(self.n, self.fft_index, **constraint)
        self.smoothness = smoothness

    def create_index(self) -> np.ndarray:
        # Determine min and max values for indices based on FFT indexing or regular indexing
        if self.fft_index:
            max_vals = (self.n - 1) // 2
            min_vals = max_vals - self.n + 1
        else:
            min_vals = np.zeros_like(self.n)
            max_vals = self.n - 1

        # Generate equispaced indices
        indices = create_equispaced_array(self.anchor, min_vals, max_vals, self.steps)

        # Apply the constraint filter if provided
        if self.constraint:
            indices = np.asarray([idx for idx in indices if self.constraint(idx)])

        # Wrap indices for FFT indexing
        if self.fft_index:
            indices %= self.n

        return indices

    def split(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        roll_to_origin: bool = True,
    ) -> np.ndarray:
        assert len(x.shape) == indices.shape[1], "Dimensions don't match."
        radius = self.steps / 2
        func = lambda shift: np.linalg.norm(shift / radius, ord=np.inf, axis=-1)
        out = split(
            x,
            indices,
            func=func,
            eps=self.smoothness,
            fft_index=self.fft_index,
            roll_to_origin=roll_to_origin,
        )
        return out


class DiamondScheme(BasicScheme):

    def __init__(
        self,
        n: Sequence[int],
        steps: Sequence[int],
        anchor: Sequence[int] = None,
        fft_index: bool = False,
        constraint: dict = {},
        smoothness: float = 0.2,
    ):
        self.n = np.asarray(n)
        self.steps = np.asarray(steps)
        if anchor is not None:
            self.anchor = np.asarray(anchor)
        else:
            self.anchor = np.zeros_like(self.n)
        self.fft_index = fft_index
        self.constraint = create_constraint(self.n, self.fft_index, **constraint)
        self.smoothness = smoothness

        # Define the diamond constraint
        self.diamond_constraint = (
            lambda idx: np.sum(((idx - self.anchor) // self.steps)) % len(self.n) == 0
        )

    def create_index(self) -> np.ndarray:
        # Determine min and max values for indices based on FFT indexing or regular indexing
        if self.fft_index:
            max_vals = (self.n - 1) // 2
            min_vals = max_vals - self.n + 1
        else:
            min_vals = np.zeros_like(self.n)
            max_vals = self.n - 1

        # Generate equispaced indices
        indices = create_equispaced_array(self.anchor, min_vals, max_vals, self.steps)

        # Apply the diamond filter
        indices = np.asarray([idx for idx in indices if self.diamond_constraint(idx)])

        # Apply the constraint filter if provided
        if self.constraint:
            indices = np.asarray([idx for idx in indices if self.constraint(idx)])

        # Wrap indices for FFT indexing
        if self.fft_index:
            indices %= self.n

        return indices

    def split(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        roll_to_origin: bool = True,
    ) -> np.ndarray:
        assert len(x.shape) == indices.shape[1], "Dimensions don't match."
        radius = self.steps
        func = lambda shift: np.linalg.norm(shift / radius, ord=1, axis=-1)
        out = split(
            x,
            indices,
            func=func,
            eps=self.smoothness,
            fft_index=self.fft_index,
            roll_to_origin=roll_to_origin,
        )
        return out


class CircleScheme(BasicScheme):

    def __init__(
        self,
        n: Sequence[int],
        l2_ratio: float,
        num_angles: int,
        anchor: float = 0.0,
        fft_index: bool = True,
        constraint: dict = {},
        smoothness: float = 0.2,
    ):
        self.n = np.asarray(n)
        self.l2_ratio = l2_ratio
        self.num_angles = num_angles
        self.anchor = anchor
        self.fft_index = fft_index
        self.constraint = create_constraint(self.n, self.fft_index, **constraint)
        self.smoothness = smoothness

        assert self.fft_index
        assert np.all(self.n % 2 == 1)
        assert self.num_angles % 2 == 0

    def create_index(self) -> np.ndarray:
        # Create half-circle indices
        m = self.num_angles // 2
        angles = np.linspace(0, np.pi, m, endpoint=False) + self.anchor
        indices = self.l2_ratio * np.array([np.cos(angles), np.sin(angles)]).T
        indices = np.round(indices * (self.n // 2)).astype(int)

        # Apply the constraint filter if provided
        if self.constraint:
            indices = np.asarray([idx for idx in indices if self.constraint(idx)])

        # Fill the other half circle
        indices = np.concatenate([indices, self.n - indices], axis=0)

        # Wrap indices for FFT indexing
        if self.fft_index:
            indices %= self.n

        return indices


class CustomEllipseSplitingScheme(BasicScheme):

    def __init__(
        self,
        n: Sequence[int],
        sigmas: Sequence[Sequence[float]],
        fft_index: bool = False,
        constraint: dict = {},
        smoothness: float = 0.2,
    ):
        self.n = np.asarray(n)
        self.sigmas = np.asarray(sigmas)
        self.fft_index = fft_index
        self.constraint = constraint
        self.smoothness = smoothness

    def split(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        roll_to_origin: bool = True,
    ) -> np.ndarray:
        assert len(x.shape) == indices.shape[1], "Dimensions don't match."
        out = []
        for index, sigma in zip(indices, self.sigmas):
            func = lambda shift: np.linalg.norm(shift / sigma, ord=2, axis=-1)
            single_out = split(
                x,
                np.asarray([index]),
                func=func,
                eps=self.smoothness,
                fft_index=self.fft_index,
                roll_to_origin=roll_to_origin,
            )[0]
            out.append(single_out)
        out = np.asarray(out)
        return out


class CustomScheme(BasicScheme):
    def __init__(
        self,
        n: Sequence[int],
        indices: Sequence[Sequence[int]],
        fft_index: bool = False,
    ):
        self.n = np.asarray(n)
        self.indices = np.asarray(indices)
        self.fft_index = fft_index

    def create_index(self) -> np.ndarray:
        return self.indices

    def split(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        roll_to_origin: bool = True,
        **kwargs,
    ) -> np.ndarray:
        assert indices.shape[0] == 1, "Must be only one point"
        if roll_to_origin:
            x = np.roll(x, -indices[0], axis=range(len(self.n)))
        return np.asarray([x])


def get_scheme(method: str, **kwargs) -> BasicScheme:
    if method == "equispace":
        return EquispaceScheme(**kwargs)
    elif method == "diamond":
        return DiamondScheme(**kwargs)
    elif method == "circle":
        return CircleScheme(**kwargs)
    elif method == "ellipse":
        return CustomEllipseSplitingScheme(**kwargs)
    elif method == "custom":
        return CustomScheme(**kwargs)
    else:
        ValueError(method)


# Tests
def test_EquispaceScheme():
    n = (9, 7)
    steps = (2, 2)
    scheme = EquispaceScheme(n, steps)
    out = scheme.create_index()
    out_true = np.array([[x, y] for x in range(0, 9, 2) for y in range(0, 7, 2)])
    assert np.all(out == out_true)
    v = np.arange(n[0] * n[1]).reshape(n)
    indices = np.array([[2, 2], [8, 4]])
    scheme.split(v, indices).round().astype(int)


def test_DiamondScheme():
    n = (9, 7)
    steps = (1, 1)
    constraint = {"l2_ratio": [0.1, 0.5]}
    scheme = DiamondScheme(n, steps, fft_index=True, constraint=constraint)
    out = scheme.create_index()
    out_true = np.array([[7, 0], [8, 6], [8, 1], [1, 6], [1, 1], [2, 0]])
    assert np.all(out == out_true)

    constraint = {"l2_ratio": [0.0, 1.0]}
    scheme = DiamondScheme(n, steps, fft_index=True, constraint=constraint)
    out = scheme.create_index()
    v = np.zeros(n, dtype=int)
    v[tuple(out.T)] = 1
    print(fft.fftshift(v))

    v = np.arange(n[0] * n[1]).reshape(n)
    indices = np.array([[2, 2], [4, 2]])
    scheme.split(v, indices).round().astype(int)


def test_CircleScheme():
    n = (9, 7)
    l2_ratio = 0.5
    num_angles = 8
    scheme = CircleScheme(n, l2_ratio, num_angles)
    out = scheme.create_index()
    out_true = np.array(
        [[2, 0], [1, 1], [0, 2], [8, 1], [7, 0], [8, 6], [0, 5], [1, 6]]
    )
    assert np.all(out == out_true)

    l2_ratio = 0.9
    scheme = CircleScheme(n, l2_ratio, num_angles)
    out = scheme.create_index()
    v = np.zeros(n, dtype=int)
    v[tuple(out.T)] = 1
    print(fft.fftshift(v).T)


def test_CustomEllipseSplitingScheme():
    n = (21, 21)
    sigmas = [[4, 2], [6, 3]]
    scheme = CustomEllipseSplitingScheme(n, sigmas, fft_index=True)
    v = np.ones(n, dtype=int)
    indices = np.array([[5, 0], [10, 8]])
    out = scheme.split(v, indices, smoothness=0.0).round().astype(int)
    for x in out:
        print(fft.fftshift(x).T)
