import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pipeline.loader import XRDData
from pipeline.fitter import XRDFitter, FitResult


def make_synthetic_xrd(
    peak_centers=(26.65, 31.40),
    fwhm=0.20,
    n_points=300,
    two_theta_range=(20.0, 40.0),
    noise_level=0.5,
) -> XRDData:
    """Create a synthetic XRD pattern with known Gaussian peaks."""
    x = np.linspace(two_theta_range[0], two_theta_range[1], n_points)
    y = np.full_like(x, 3.0)  # flat background
    for center in peak_centers:
        sigma = fwhm / 2.3548
        y += 100.0 * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    rng = np.random.default_rng(42)
    y += rng.normal(0, noise_level, size=len(x))
    y = np.clip(y, 0, None)
    y = y / y.max() * 100.0

    df = pd.DataFrame({"two_theta": x, "intensity": y})
    return XRDData(
        name="synthetic",
        path=Path("synthetic.txt"),
        metadata={},
        df=df,
        wavelength=1.54056,
    )


class TestXRDFitter:
    def test_fit_returns_fitresult(self):
        xrd = make_synthetic_xrd()
        result = XRDFitter().fit_sample(xrd)
        assert isinstance(result, FitResult)

    def test_dominant_peak_center_accurate(self):
        xrd = make_synthetic_xrd(peak_centers=(26.65,), fwhm=0.20)
        result = XRDFitter().fit_sample(xrd)
        assert abs(result.dominant_peak["center"] - 26.65) < 0.15

    def test_r_squared_noiseless(self):
        xrd = make_synthetic_xrd(noise_level=0.0)
        result = XRDFitter().fit_sample(xrd)
        assert result.r_squared > 0.99

    def test_two_peaks_detected(self):
        xrd = make_synthetic_xrd(peak_centers=(26.65, 31.40))
        result = XRDFitter().fit_sample(xrd)
        assert result.n_peaks >= 2

    def test_dominant_peak_dict_keys(self):
        xrd = make_synthetic_xrd()
        result = XRDFitter().fit_sample(xrd)
        assert set(result.dominant_peak.keys()) == {"center", "fwhm", "amplitude", "eta"}

    def test_fwhm_positive(self):
        xrd = make_synthetic_xrd()
        result = XRDFitter().fit_sample(xrd)
        assert result.dominant_peak["fwhm"] > 0

    def test_aic_bic_finite(self):
        xrd = make_synthetic_xrd()
        result = XRDFitter().fit_sample(xrd)
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

    def test_wavelength_propagated(self):
        xrd = make_synthetic_xrd()
        result = XRDFitter().fit_sample(xrd)
        assert result.wavelength == 1.54056

    def test_fit_batch_returns_dict(self):
        xrd = make_synthetic_xrd()
        results = XRDFitter().fit_batch({"synthetic": xrd}, progress=False)
        assert "synthetic" in results
        assert isinstance(results["synthetic"], FitResult)

    def test_fit_batch_handles_exception_gracefully(self):
        # Pass an XRDData with empty df — should not crash, just warn
        import pandas as pd
        bad_xrd = XRDData(
            name="bad", path=Path("bad.txt"), metadata={},
            df=pd.DataFrame({"two_theta": [], "intensity": []}),
            wavelength=1.54056,
        )
        results = XRDFitter().fit_batch({"bad": bad_xrd}, progress=False)
        # Either succeeded or was caught — no crash
        assert True
