import pytest
import numpy as np
from autoxrd.loader import XRDLoader, XRDData


class TestXRDLoader:
    def test_load_returns_xrddata(self, synthetic_rruff_file):
        xrd = XRDLoader.load(synthetic_rruff_file)
        assert isinstance(xrd, XRDData)

    def test_dataframe_columns(self, synthetic_rruff_file):
        xrd = XRDLoader.load(synthetic_rruff_file)
        assert list(xrd.df.columns) == ["two_theta", "intensity"]

    def test_intensity_normalized_to_100(self, synthetic_rruff_file):
        xrd = XRDLoader.load(synthetic_rruff_file)
        assert abs(xrd.df["intensity"].max() - 100.0) < 1e-6

    def test_wavelength_read_from_header(self, synthetic_rruff_file):
        xrd = XRDLoader.load(synthetic_rruff_file)
        assert abs(xrd.wavelength - 1.54056) < 1e-6

    def test_wavelength_defaults_when_absent(self, synthetic_rruff_no_wavelength):
        xrd = XRDLoader.load(synthetic_rruff_no_wavelength)
        assert abs(xrd.wavelength - XRDLoader.DEFAULT_WAVELENGTH) < 1e-6

    def test_metadata_captured(self, synthetic_rruff_file):
        xrd = XRDLoader.load(synthetic_rruff_file)
        assert "NAMES" in xrd.metadata
        assert "RRUFFID" in xrd.metadata

    def test_name_defaults_to_stem(self, synthetic_rruff_file):
        xrd = XRDLoader.load(synthetic_rruff_file)
        assert xrd.name == "Synthetic__R000000"

    def test_name_override(self, synthetic_rruff_file):
        xrd = XRDLoader.load(synthetic_rruff_file, name="MyName")
        assert xrd.name == "MyName"

    def test_data_rows_parsed(self, synthetic_rruff_file):
        xrd = XRDLoader.load(synthetic_rruff_file)
        assert len(xrd.df) == 200

    def test_load_directory_skips_download_script(self, tmp_path):
        # download_samples.py should be skipped
        (tmp_path / "download_samples.py").write_text("# script")
        (tmp_path / "Quartz__R040031.txt").write_text("##NAMES=Quartz\n26.0, 100.0\n")
        result = XRDLoader.load_directory(tmp_path)
        assert "Quartz__R040031" in result
        assert len(result) == 1  # download script not included
