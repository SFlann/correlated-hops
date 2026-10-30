"""Shared validation for discrete public API parameters."""

from operator import index

import numpy as np


def integer(value: object, name: str, *, minimum: int = 1) -> int:
    """Accept Python/NumPy integers, rejecting floats and booleans consistently."""
    message = f"{name} must be an integer >= {minimum}"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        result = index(value)
    except TypeError:
        raise ValueError(message) from None
    if result < minimum:
        raise ValueError(message)
    return result
