"""
Extended transition network symbolizers.

SAX (Symbolic Aggregate approXimation) and entropy-maximizing binning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_positive_int, validate_series
from ..api import TransitionNetwork


def sax_symbolize(
    x: NDArray[np.float64],
    n_bins: int = 8,
    word_size: int = 3,
) -> NDArray[np.int32]:
    """
    Symbolic Aggregate approXimation (SAX) of a time series.

    Parameters
    ----------
    x : array (n,)
    n_bins : int
        Alphabet size.
    word_size : int
        PAA segment size (series length should be divisible).

    Returns
    -------
    symbols : array (n_words,)
        Integer symbols per SAX word.
    """
    x = validate_series(x, "sax_symbolize")
    n_bins = validate_positive_int("n_bins", n_bins, minimum=2)
    word_size = validate_positive_int("word_size", word_size)

    n = len(x)
    n_words = n // word_size
    if n_words == 0:
        raise ValueError("Series too short for given word_size")

    truncated = x[: n_words * word_size].reshape(n_words, word_size)
    paa = truncated.mean(axis=1)

    # Gaussian breakpoints for equal-frequency bins
    breakpoints = np.linspace(-np.inf, np.inf, n_bins + 1)[1:-1]
    if np.std(paa) > 0:
        z = (paa - paa.mean()) / paa.std()
    else:
        z = np.zeros_like(paa)
    return np.digitize(z, breakpoints).astype(np.int32)


def entropy_max_symbolize(
    x: NDArray[np.float64],
    n_bins: int = 8,
) -> NDArray[np.int32]:
    """
    Equal-frequency (entropy-maximizing) symbolization.

    Assigns symbols so each bin has approximately equal count.
    """
    x = validate_series(x, "entropy_max_symbolize")
    n_bins = validate_positive_int("n_bins", n_bins, minimum=2)
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(x, quantiles)
    return np.digitize(x, edges).astype(np.int32)


def sax_transition_network(
    x: NDArray[np.float64],
    n_bins: int = 8,
    word_size: int = 3,
    output: str = "edges",
) -> tuple[TransitionNetwork, NDArray[np.int32]]:
    """
    Build a transition network on SAX symbols.

    Uses equal-width transitions between consecutive SAX words mapped to
    ordinal symbols, via ``TransitionNetwork(symbolizer='equal_freq')`` on
    the SAX symbol sequence.

    Returns
    -------
    builder : TransitionNetwork
    symbols : SAX symbol sequence
    """
    symbols = sax_symbolize(x, n_bins=n_bins, word_size=word_size).astype(np.float64)
    builder = TransitionNetwork(
        symbolizer="equal_freq",
        order=1,
        bins=n_bins,
        output=output,
    ).build(symbols)
    return builder, symbols.astype(np.int32)
