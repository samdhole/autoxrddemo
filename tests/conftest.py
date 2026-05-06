import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


@pytest.fixture
def synthetic_rruff_file(tmp_path):
    """Create a synthetic RRUFF-format .txt file with 2 known peaks."""
    content = """##NAMES=Synthetic Quartz
##RRUFFID=R000000
##X-RAY WAVELENGTH=1.54056
##STATUS=
26.65, 100.0
27.00, 80.0
27.35, 40.0
31.00, 20.0
31.40, 90.0
31.80, 25.0
"""
    filepath = tmp_path / "Synthetic__R000000.txt"
    filepath.write_text(content)
    return filepath


@pytest.fixture
def synthetic_rruff_no_wavelength(tmp_path):
    """RRUFF file without wavelength field — should default to 1.54056."""
    content = """##NAMES=NoWavelength Mineral
##RRUFFID=R000001
20.0, 50.0
25.0, 100.0
30.0, 60.0
"""
    filepath = tmp_path / "NoWavelength__R000001.txt"
    filepath.write_text(content)
    return filepath
