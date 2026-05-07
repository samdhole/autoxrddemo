import pytest
import numpy as np
import pandas as pd
from pathlib import Path


def _gaussian(x, center, amplitude, fwhm):
    sigma = fwhm / 2.3548
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


@pytest.fixture
def synthetic_rruff_file(tmp_path):
    """Synthetic RRUFF-format file with 2 resolved Gaussian peaks (~200 pts, 20-40 deg)."""
    x = np.linspace(20.0, 40.0, 200)
    y = (
        5.0  # flat background
        + _gaussian(x, center=26.65, amplitude=100.0, fwhm=0.20)
        + _gaussian(x, center=31.40, amplitude=80.0, fwhm=0.18)
    )
    y = y / y.max() * 100.0

    lines = ["##NAMES=Synthetic Quartz\n", "##RRUFFID=R000000\n", "##X-RAY WAVELENGTH=1.54056\n"]
    for xi, yi in zip(x, y):
        lines.append(f"{xi:.4f}, {yi:.4f}\n")

    filepath = tmp_path / "Synthetic__R000000.txt"
    filepath.write_text("".join(lines), encoding="utf-8")
    return filepath


@pytest.fixture
def synthetic_rruff_no_wavelength(tmp_path):
    """RRUFF file without wavelength header — loader should default to 1.54056 A."""
    x = np.linspace(20.0, 35.0, 150)
    y = 5.0 + _gaussian(x, center=25.0, amplitude=100.0, fwhm=0.25)
    y = y / y.max() * 100.0

    lines = ["##NAMES=NoWavelength Mineral\n", "##RRUFFID=R000001\n"]
    for xi, yi in zip(x, y):
        lines.append(f"{xi:.4f}, {yi:.4f}\n")

    filepath = tmp_path / "NoWavelength__R000001.txt"
    filepath.write_text("".join(lines), encoding="utf-8")
    return filepath
