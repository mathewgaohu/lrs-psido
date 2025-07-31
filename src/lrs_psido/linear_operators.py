"""Interface of Pseudo-differential operators as scipy linear operaters."""

import numpy as np
from scipy import fft
from scipy.sparse.linalg import LinearOperator


class PDOLowRankSymbol(LinearOperator):
    """Pseudo-differential operator with a low-rank symbol approximation.

    The symbol is approximated as:
        S(x, xi) ~ sum_{k=1}^r C_k(x) R_k(xi)
    The operator applies the transformation:
        A u = sum_{k=1}^r C_k F^{-1} (R_k * F u)
    where F is the Fourier transform.
    """

    def __init__(
        self,
        C: np.ndarray,
        R: np.ndarray,
        dtype: str | type = np.complex128,
        fft_workers: int = -1,
    ):
        """Initialize the pseudo-differential operator.

        Args:
            C (np.ndarray): Coefficients of shape (r, n1, n2, ...).
            R (np.ndarray): Fourier multipliers of shape (r, n1, n2, ...).
            dtype (str | type, optional): Data type. Defaults to np.complex128.
            fft_workers (int, optional): Number of workers for FFT. Defaults to -1.
        """
        C = np.asarray(C, dtype=dtype)
        R = np.asarray(R, dtype=dtype)
        assert C.shape == R.shape, "C and R must have the same shape."

        self.r, *self.n = C.shape  # Rank r and spatial dimensions
        self.C = C
        self.R = R
        self.fft_workers = fft_workers

        # Define operator shape based on the flattened size of spatial dimensions
        shape = (np.prod(self.n), np.prod(self.n))
        super().__init__(shape=shape, dtype=dtype)

        # Track function calls for debugging
        self.ncalls_fwd = 0
        self.ncalls_adj = 0

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply the forward operator A * x."""
        self.ncalls_fwd += 1
        x = x.reshape(self.n)

        out = (
            fft.ifftn(
                fft.fftn(x, s=self.n, workers=self.fft_workers) * self.R,
                s=self.n,
                workers=self.fft_workers,
            )
            * self.C
        )
        return np.sum(out, axis=0, dtype=self.dtype).ravel()

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Apply the adjoint operator A^H * x."""
        self.ncalls_adj += 1
        x = x.reshape(self.n)

        out = fft.ifftn(
            fft.fftn(x * self.C.conj(), s=self.n, workers=self.fft_workers)
            * self.R.conj(),
            s=self.n,
            workers=self.fft_workers,
        )
        return np.sum(out, axis=0, dtype=self.dtype).ravel()

    @property
    def real(self):
        """Return the real part of the operator."""
        if not hasattr(self, "_real_op"):
            self._real_op = RealPartLinearOperator(self)
        return self._real_op


class RealPartLinearOperator(LinearOperator):
    """Linear operator that extracts the real part of another linear operator.

    Given a complex-valued operator A, this class defines a new operator
    that represents only the real part of A.
    """

    def __init__(self, A: LinearOperator):
        """Initialize the real-part operator.

        Args:
            A (LinearOperator): The original linear operator (possibly complex-valued).
        """
        if np.iscomplexobj(A.dtype):
            real_dtype = np.float32 if np.dtype(A.dtype) == np.complex64 else np.float64
        else:
            raise NotImplementedError("A is already real.")

        super().__init__(shape=A.shape, dtype=real_dtype)
        self.A = A

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute the matrix-vector product, keeping only the real part.

        Args:
            x (np.ndarray): Input vector.

        Returns:
            np.ndarray: Real part of A @ x.
        #"""
        if np.iscomplexobj(x):
            return self._matvec(x.real) + 1j * self._matvec(x.imag)
        Ax = self.A @ x
        return np.real(Ax)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Compute the adjoint matrix-vector product, keeping only the real part.

        Args:
            x (np.ndarray): Input vector.

        Returns:
            np.ndarray: Real part of A^H @ x.
        """
        if np.iscomplexobj(x):
            return self._rmatvec(x.real) + 1j * self._rmatvec(x.imag)
        Ax = self.A.H @ x
        return np.real(Ax)


class RealPDOLowRankSymbol(LinearOperator):
    """Pseudo-differential operator with a low-rank symbol approximation.

    The symbol is approximated as:
        S(x, xi) ~ sum_{k=1}^r C_k(x) R_k(xi).
    where:
    - `C` is real-valued.
    - `R` is central conjugate symmetric, enabling efficient computation via rFFT.

    The operator is applied as:
        A u = sum_{k=1}^{r} C_k * F^{-1} (R_k * F u),
    where `F` is the real Fourier transform (rFFT).
    """

    def __init__(
        self,
        C: np.ndarray,
        R: np.ndarray,
        dtype: str | type = np.float64,
        fft_workers: int = -1,
    ):
        """Initializes the real-valued pseudo-differential operator.

        Args:
            C (np.ndarray): Coefficient tensor of shape (r, n1, n2, ...), real-valued.
            R (np.ndarray): Frequency-domain tensor of shape (r, n1, n2, ...), central conjugate symmetric.
            dtype (np.dtype, optional): Data type for computations. Defaults to np.float64.
            fft_workers (int, optional): Number of parallel workers for FFT computations. Defaults to -1 (auto).
        """
        # Ensure dtype compatibility
        dtype: np.dtype = np.dtype(dtype)
        if dtype.type == np.float32:
            cdtype = np.complex64
        elif dtype.type == np.float64:
            cdtype = np.complex128
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        C = np.asarray(C, dtype=dtype)
        R = np.asarray(R, dtype=cdtype)
        assert C.shape[:-1] == R.shape[:-1], "C and R must have matching dimensions."

        # Handle different shapes of R (full vs half-spectrum storage)
        expected_last_dim = C.shape[-1] // 2 + 1
        if R.shape[-1] == C.shape[-1]:
            Rh = R[..., :expected_last_dim]
        elif R.shape[-1] == expected_last_dim:
            Rh = R
        else:
            raise ValueError(
                f"Expected last dimension of R to be {C.shape[-1]} or {expected_last_dim}, got {R.shape[-1]}"
            )

        self.r, *self.n = C.shape  # Rank r and spatial dimensions
        self.C = C
        self.Rh = Rh
        self.fft_workers = fft_workers

        # Define operator shape based on the flattened size of spatial dimensions
        shape = (np.prod(self.n), np.prod(self.n))
        super().__init__(shape=shape, dtype=dtype)

        # Track function calls for debugging/performance monitoring
        self.ncalls_fwd = 0
        self.ncalls_adj = 0

    @property
    def R(self) -> np.ndarray:
        """Reconstructs the full frequency-domain representation of R."""
        return fft.fftn(fft.irfftn(self.Rh, s=self.n), s=self.n)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Forward application of the pseudo-differential operator.

        Args:
            x (np.ndarray): Input vector of size matching the operator.

        Returns:
            np.ndarray: Output after applying the operator.
        """
        if np.iscomplexobj(x):
            return self._matvec(x.real) + 1j * self._matvec(x.imag)

        self.ncalls_fwd += 1
        x = x.reshape(self.n)

        out = (
            fft.irfftn(
                fft.rfftn(x, s=self.n, workers=self.fft_workers) * self.Rh,
                s=self.n,
                workers=self.fft_workers,
            )
            * self.C
        )
        return np.sum(out, axis=0, dtype=self.dtype).ravel()

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Adjoint application of the pseudo-differential operator.

        Args:
            x (np.ndarray): Input vector.

        Returns:
            np.ndarray: Output after applying the adjoint of the operator.
        """
        if np.iscomplexobj(x):
            return self._rmatvec(x.real) + 1j * self._rmatvec(x.imag)

        self.ncalls_adj += 1
        x = x.reshape(self.n)

        out = fft.irfftn(
            fft.rfftn(x * self.C, s=self.n, workers=self.fft_workers) * self.Rh.conj(),
            s=self.n,
            workers=self.fft_workers,
        )
        return np.sum(out, axis=0, dtype=self.dtype).ravel()
