#!/usr/bin/env python3
"""Download UCR datasets and write bundled ``.npz`` files for offline use."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[1] / "ts2net" / "datasets" / "data" / "ucr"
_NAMES = ("GunPoint", "ItalyPowerDemand", "Coffee")


def main() -> None:
    from aeon.datasets import load_classification

    _DATA.mkdir(parents=True, exist_ok=True)
    for name in _NAMES:
        for split in ("train", "test"):
            X, y = load_classification(name, split=split)
            X = np.asarray(X.squeeze(), dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = np.asarray(y)
            path = _DATA / f"{name}_{split}.npz"
            np.savez_compressed(path, X=X, y=y)
            print(f"wrote {path} shape={X.shape}")


if __name__ == "__main__":
    main()
