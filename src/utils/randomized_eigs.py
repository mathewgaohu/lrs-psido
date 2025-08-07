import numpy as np
from scipy.sparse.linalg import LinearOperator


def randomized_eigs(
    A: LinearOperator,
    B: LinearOperator,
    Binv: LinearOperator,
    rank: int,
    oversampling: int,
) -> tuple[np.ndarray, np.ndarray]: ...
