import numpy as np
from typing import Tuple, List, Optional


def unitary_to_cache_key(unitary: np.ndarray, precision: int = 10) -> Tuple[Tuple[float, float], ...]:
    """Convert a 4x4 unitary matrix into a hashable tuple, rounded to the specified precision."""
    rounded = np.round(unitary.real, precision) + 1j * np.round(unitary.imag, precision)
    return tuple((val.real, val.imag) for val in rounded.flatten())


