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

    def test_malformed_comment_only_file_raises_cleanly(self, tmp_path):
        path = tmp_path / "comment_only.txt"
        path.write_text("##NAMES=No data\n# all comments\n")

        with pytest.raises(ValueError, match="No XRD data rows found"):
            XRDLoader.load(path)

    def test_load_directory_propagates_malformed_comment_only_file(self, tmp_path):
        (tmp_path / "good.txt").write_text("##NAMES=Good\n26.0, 100.0\n")
        (tmp_path / "bad.txt").write_text("##NAMES=Bad\n# all comments\n")

        with pytest.raises(ValueError, match="No XRD data rows found"):
            XRDLoader.load_directory(tmp_path)

    def test_load_directory_returns_one_entry_per_txt_file(self, tmp_path):
        (tmp_path / "A.txt").write_text("##NAMES=A\n26.0, 100.0\n")
        (tmp_path / "B.txt").write_text("##NAMES=B\n27.0, 50.0\n")
        (tmp_path / "ignore.csv").write_text("28.0, 25.0\n")

        result = XRDLoader.load_directory(tmp_path)

        assert set(result) == {"A", "B"}
        assert len(result) == 2

    def test_resampling_enforces_max_points_for_just_over_limit(self, tmp_path):
        path = tmp_path / "dense.txt"
        rows = ["##NAMES=Dense\n"]
        for i in range(XRDLoader.MAX_POINTS + 1):
            rows.append(f"{20.0 + i * 0.01:.4f}, {float(i + 1):.1f}\n")
        path.write_text("".join(rows), encoding="utf-8")

        xrd = XRDLoader.load(path)

        assert len(xrd.df) <= XRDLoader.MAX_POINTS
