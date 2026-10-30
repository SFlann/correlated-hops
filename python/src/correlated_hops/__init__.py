"""Correlated HOPS in a local, factorially scaled hierarchy basis."""

from .bath import Bath, stationary_noise
from .models import spin_chain, spin_operators
from .observables import concurrence, connected, expectation, partial_trace
from .solver import Model, Result, solve

__version__ = "0.1.0"
__all__ = [
    "Bath",
    "stationary_noise",
    "Model",
    "Result",
    "solve",
    "spin_chain",
    "spin_operators",
    "expectation",
    "connected",
    "concurrence",
    "partial_trace",
]
