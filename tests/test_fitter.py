import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from autoxrd.loader import XRDData
from autoxrd.fitter import XRDFitter, FitResult


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
        # Failed sample must be excluded from results, not stored as None
        assert "bad" not in results

    def test_all_peaks_length_matches_n_peaks(self):
        xrd = make_synthetic_xrd(peak_centers=(26.65, 31.40))
        result = XRDFitter().fit_sample(xrd)
        assert len(result.all_peaks) == result.n_peaks

    def test_all_peaks_sorted_by_center(self):
        xrd = make_synthetic_xrd(peak_centers=(26.65, 31.40))
        result = XRDFitter().fit_sample(xrd)
        centers = [p["center"] for p in result.all_peaks]
        assert centers == sorted(centers)

    def test_relative_intensity_max_is_100(self):
        xrd = make_synthetic_xrd(peak_centers=(26.65, 31.40))
        result = XRDFitter().fit_sample(xrd)
        if result.all_peaks:
            assert max(p["relative_intensity"] for p in result.all_peaks) == pytest.approx(100.0, abs=0.1)

    def test_all_peaks_have_required_keys(self):
        xrd = make_synthetic_xrd()
        result = XRDFitter().fit_sample(xrd)
        for p in result.all_peaks:
            assert {"center", "fwhm", "amplitude", "eta", "d_spacing", "relative_intensity"} <= p.keys()

    def test_synthetic_quartz_recall(self):
        """DoG detection must recover ≥14 of 15 known reflections in synthetic quartz."""
        from autoxrd.loader import XRDLoader
        repo_root = Path(__file__).resolve().parent.parent
        f = repo_root / "data" / "synthetic_quartz" / "Quartz_Anneal_01.txt"
        if not f.exists():
            pytest.skip("synthetic_quartz dataset not present")
        truth = [20.86, 26.65, 36.54, 39.47, 40.29, 42.45, 45.79, 50.14,
                 54.87, 55.33, 59.96, 64.02, 67.74, 68.14, 73.47]
        xrd = XRDLoader.load(f)
        result = XRDFitter().fit_sample(xrd)
        detected = [p["center"] for p in result.all_peaks]
        recall = sum(1 for t in truth if any(abs(d - t) < 0.3 for d in detected))
        false_pos = sum(1 for d in detected if not any(abs(d - t) < 0.3 for t in truth))
        assert recall >= 14, f"recall={recall}/15; detected={[round(c,2) for c in detected]}"
        assert false_pos <= 2, f"false positives={false_pos}; detected={[round(c,2) for c in detected]}"

    def test_eta_low_on_pure_gaussian(self):
        """Synthetic data uses pure Gaussian peaks. Fitter η (Lorentzian fraction) on the
        dominant peak should converge near 0; η > 0.3 would signal a model/data mismatch."""
        xrd = make_synthetic_xrd(noise_level=0.0)
        result = XRDFitter().fit_sample(xrd)
        assert result.dominant_peak["eta"] < 0.3, f"η={result.dominant_peak['eta']:.3f} too high for pure Gaussian"

    def test_amplitude_threshold_relative_to_y_max(self):
        """Amplitude floor scales with spectrum's intensity range, so it works on
        non-normalized data too. Doubling y also doubles the threshold —
        n_peaks must (a) be > 0 in both cases (sanity), (b) match between cases."""
        xrd_norm = make_synthetic_xrd(noise_level=0.0)
        xrd_scaled = XRDData(
            name=xrd_norm.name, path=xrd_norm.path, metadata=xrd_norm.metadata,
            df=pd.DataFrame({"two_theta": xrd_norm.df["two_theta"].values,
                            "intensity": xrd_norm.df["intensity"].values * 1000.0}),
            wavelength=xrd_norm.wavelength,
        )
        n_norm = XRDFitter().fit_sample(xrd_norm).n_peaks
        n_scaled = XRDFitter().fit_sample(xrd_scaled).n_peaks
        assert n_norm > 0, "test is vacuous: normalized fit found no peaks"
        assert n_scaled > 0, "test is vacuous: scaled fit found no peaks"
        assert n_norm == n_scaled, f"scaling broke the amp threshold: {n_norm} vs {n_scaled}"


class TestWarmStart:
    def test_warm_start_batch_completes(self, synthetic_rruff_file):
        from autoxrd.loader import XRDLoader
        xrd = XRDLoader.load(synthetic_rruff_file)
        samples = {"s1": xrd, "s2": xrd, "s3": xrd}
        fitter = XRDFitter()
        results_cold = fitter.fit_batch(samples, progress=False, warm_start=False)
        results_warm = fitter.fit_batch(samples, progress=False, warm_start=True)
        assert set(results_warm.keys()) == set(results_cold.keys())

    def test_warm_start_same_peak_count(self, synthetic_rruff_file):
        from autoxrd.loader import XRDLoader
        xrd = XRDLoader.load(synthetic_rruff_file)
        samples = {"s1": xrd, "s2": xrd}
        fitter = XRDFitter()
        cold = fitter.fit_batch(samples, progress=False, warm_start=False)
        warm = fitter.fit_batch(samples, progress=False, warm_start=True)
        assert warm["s2"].n_peaks == cold["s2"].n_peaks

    def test_fit_one_roi_accepts_warm_sigma(self):
        fitter = XRDFitter()
        x = np.linspace(26.0, 27.3, 65)
        sigma_true = 0.08
        y = np.exp(-0.5 * ((x - 26.65) / sigma_true) ** 2) * 80.0 + 1.0
        peak_idx = int(np.argmax(y))
        result_cold = fitter._fit_one_roi(x, y, peak_idx, warm_sigma=None)
        result_warm = fitter._fit_one_roi(x, y, peak_idx, warm_sigma=0.08)
        assert result_warm["fwhm"] > 0
        assert abs(result_warm["fwhm"] - result_cold["fwhm"]) < 0.05

    def test_warm_start_false_is_default(self):
        import inspect
        sig = inspect.signature(XRDFitter.fit_batch)
        assert sig.parameters["warm_start"].default is False
