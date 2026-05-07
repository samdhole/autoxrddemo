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
        assert result.all_peaks, "test is vacuous: synthetic fit found no peaks"
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

    def test_invalid_detected_peak_indices_are_skipped(self, monkeypatch):
        xrd = make_synthetic_xrd()
        fitter = XRDFitter()

        def bad_detect_peaks(x, y):
            return np.array([len(x), -1]), np.zeros_like(y), 0.1

        monkeypatch.setattr(fitter, "_detect_peaks", bad_detect_peaks)

        result = fitter.fit_sample(
            xrd,
            warm_peaks=[{"center": 26.65, "fwhm": 0.20, "amplitude": 100.0, "eta": 0.2}],
        )

        assert result.n_peaks == 0
        assert result.all_peaks == []
        assert np.isnan(result.dominant_peak["center"])

    def test_all_optimizer_failures_return_empty_fit_result(self, monkeypatch):
        xrd = make_synthetic_xrd()
        fitter = XRDFitter()

        def one_detected_peak(x, y):
            return np.array([int(np.argmax(y))]), np.zeros_like(y), 0.1

        def always_fail(*_args, **_kwargs):
            raise RuntimeError("optimizer failed")

        monkeypatch.setattr(fitter, "_detect_peaks", one_detected_peak)
        monkeypatch.setattr(fitter, "_fit_one_roi", always_fail)

        result = fitter.fit_sample(xrd)

        assert result.n_peaks == 0
        assert result.all_peaks == []
        assert result.r_squared == 0.0
        assert set(result.dominant_peak.keys()) == {"center", "fwhm", "amplitude", "eta"}

    def test_all_optimizer_failures_are_counted(self, monkeypatch):
        xrd = make_synthetic_xrd()
        fitter = XRDFitter()

        def two_detected_peaks(x, y):
            return np.array([int(np.argmax(y)), int(np.argmax(y)) + 5]), np.zeros_like(y), 0.1

        def always_fail(*_args, **_kwargs):
            raise RuntimeError("optimizer failed")

        monkeypatch.setattr(fitter, "_detect_peaks", two_detected_peaks)
        monkeypatch.setattr(fitter, "_fit_one_roi", always_fail)

        result = fitter.fit_sample(xrd)

        assert result.failed_peak_count == 2

    def test_dominant_peak_matches_peak_with_largest_amplitude(self):
        xrd = make_synthetic_xrd(peak_centers=(26.65, 31.40), noise_level=0.0)
        result = XRDFitter().fit_sample(xrd)
        assert result.all_peaks, "test is vacuous: synthetic fit found no peaks"

        strongest = max(result.all_peaks, key=lambda p: p["amplitude"])

        assert result.dominant_peak
        assert result.dominant_peak["center"] == pytest.approx(strongest["center"])
        assert result.dominant_peak["amplitude"] == pytest.approx(strongest["amplitude"])


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

    def test_warm_sigma_propagated_from_fit_sample(self):
        """Verify fit_sample actually passes warm_sigma to _fit_one_roi when warm_peaks match."""
        from unittest.mock import patch, call
        fitter = XRDFitter()
        x = np.linspace(26.0, 27.5, 150)
        sigma_real = 0.07
        y = np.exp(-0.5 * ((x - 26.65) / sigma_real) ** 2) * 90.0 + 2.0
        y = y / y.max() * 100.0
        df = pd.DataFrame({"two_theta": x, "intensity": y})
        xrd = XRDData(name="s", path=Path("s.txt"), metadata={}, df=df, wavelength=1.54056)

        warm_peaks = [{"center": 26.65, "fwhm": sigma_real * 2.0, "amplitude": 90.0,
                       "eta": 0.2, "d_spacing": 3.34, "relative_intensity": 100.0}]

        received_warm_sigmas = []
        original_fit_one_roi = fitter._fit_one_roi

        def spy_fit_one_roi(x, y, peak_idx, warm_sigma=None):
            received_warm_sigmas.append(warm_sigma)
            return original_fit_one_roi(x, y, peak_idx, warm_sigma=warm_sigma)

        fitter._fit_one_roi = spy_fit_one_roi
        fitter.fit_sample(xrd, warm_peaks=warm_peaks)

        assert any(ws is not None for ws in received_warm_sigmas), (
            "No warm_sigma was passed to _fit_one_roi despite matching warm_peaks"
        )

    def test_warm_start_after_failed_sample_uses_last_successful_peaks(self):
        class RecordingFitter(XRDFitter):
            def __init__(self):
                super().__init__()
                self.calls = []

            def fit_sample(self, xrd, warm_peaks=None):
                self.calls.append((xrd.name, warm_peaks))
                if xrd.name == "bad":
                    raise ValueError("synthetic failure")
                peaks = [{"center": 26.65, "fwhm": 0.20, "amplitude": 100.0,
                          "eta": 0.2, "d_spacing": 3.34, "relative_intensity": 100.0}]
                return FitResult(
                    name=xrd.name,
                    dominant_peak={"center": 26.65, "fwhm": 0.20, "amplitude": 100.0, "eta": 0.2},
                    all_peaks=peaks,
                    r_squared=0.99,
                    aic=0.0,
                    bic=0.0,
                    n_peaks=1,
                    wavelength=xrd.wavelength,
                )

        df = pd.DataFrame({"two_theta": np.linspace(20.0, 30.0, 20), "intensity": np.ones(20)})
        samples = {
            name: XRDData(name=name, path=Path(f"{name}.txt"), metadata={}, df=df, wavelength=1.54056)
            for name in ["first", "bad", "after"]
        }
        fitter = RecordingFitter()

        results = fitter.fit_batch(samples, progress=False, warm_start=True)

        assert list(results) == ["first", "after"]
        assert fitter.calls[0] == ("first", None)
        assert fitter.calls[1][0] == "bad"
        assert fitter.calls[1][1] == results["first"].all_peaks
        assert fitter.calls[2] == ("after", results["first"].all_peaks)
