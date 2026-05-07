"""DoG + ROI peak fitter for XRD spectra.

Detection: Difference of Gaussians (xrfit-style) — narrow Gaussian preserves peaks,
wide Gaussian approximates baseline; peak = where the bandpass exceeds
k_sigma × MAD(bandpass restricted to its lower half). The "k_sigma" multiplier is
applied against this lower-half MAD, which approximates noise std on the
band-passed signal — it is not a strict Gaussian-noise σ in the formal sense.

Fit: per-peak ROI (±0.25°) with single PseudoVoigt + linear local background.
~10-50 ms per peak vs minutes for global multi-peak fits.

Quality filter: drop fits with FWHM < MIN_FIT_FWHM, local R² < MIN_FIT_R2, or
integrated amplitude < MIN_FIT_AMP_FRAC × y.max() (relative to the spectrum's own
intensity range, so the threshold tracks any normalization).

Reported `r_squared` is the unweighted mean of per-peak local R² (each peak's
fit quality within its own ROI), NOT a self-referential composite R². The
display y_fit array is the sum of fitted PVs plus a smoothed baseline — used
only for plotting; it is not used to compute fit-quality metrics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from lmfit.models import PseudoVoigtModel, LinearModel

from .loader import XRDData


@dataclass
class FitResult:
    name: str
    dominant_peak: dict           # center, fwhm, amplitude, eta
    all_peaks: list               # all fitted peaks sorted by 2θ, with relative_intensity
    r_squared: float              # unweighted mean of per-peak local R²
    aic: float                    # AIC from full-spectrum residual against composite display
    bic: float                    # BIC from same; both are display-only summary stats
    n_peaks: int
    wavelength: float
    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_fit: np.ndarray = field(default_factory=lambda: np.empty(0))   # display only: baseline + sum of PVs
    baseline: np.ndarray = field(default_factory=lambda: np.empty(0))
    noise_std: float = 0.0
    detected_peak_count: int = 0  # raw DoG candidates before quality filter


class XRDFitter:
    SIGMA_SIGNAL_DEG = 0.04
    SIGMA_BASELINE_DEG = 1.5
    K_SIGMA = 4.0
    MIN_SEPARATION_DEG = 0.3
    MIN_FWHM_DETECT_DEG = 0.05
    ROI_WINDOW_DEG = 0.25

    MIN_FIT_FWHM = 0.04
    MIN_FIT_R2 = 0.5
    MIN_FIT_AMP_FRAC = 0.0025  # fraction of y.max() — equivalent to absolute 0.25 on loader-normalized 0-100 data

    def __init__(self,
                 sigma_signal_deg: float | None = None,
                 sigma_baseline_deg: float | None = None,
                 k_sigma: float | None = None,
                 roi_window_deg: float | None = None):
        self.sigma_signal_deg = sigma_signal_deg if sigma_signal_deg is not None else self.SIGMA_SIGNAL_DEG
        self.sigma_baseline_deg = sigma_baseline_deg if sigma_baseline_deg is not None else self.SIGMA_BASELINE_DEG
        self.k_sigma = k_sigma if k_sigma is not None else self.K_SIGMA
        self.roi_window_deg = roi_window_deg if roi_window_deg is not None else self.ROI_WINDOW_DEG

    def fit_sample(self, xrd: XRDData, warm_peaks: list | None = None) -> FitResult:
        x = np.asarray(xrd.df["two_theta"].values, dtype=float)
        y = np.asarray(xrd.df["intensity"].values, dtype=float)

        if len(x) < 10:
            raise ValueError(f"Sample has too few points ({len(x)}) for peak fitting")

        peak_indices, baseline, noise = self._detect_peaks(x, y)

        raw_fits = []
        composite_pv = np.zeros_like(y)
        for idx in peak_indices:
            warm_sigma = None
            try:
                idx = int(idx)
                if idx < 0 or idx >= len(x):
                    raise IndexError(f"Peak index {idx} outside sample length {len(x)}")
                if warm_peaks:
                    c_guess = float(x[idx])
                    dists = [abs(p["center"] - c_guess) for p in warm_peaks]
                    nearest = int(np.argmin(dists))
                    if dists[nearest] < 0.15:
                        warm_sigma = warm_peaks[nearest]["fwhm"] / 2.0
                fit = self._fit_one_roi(x, y, int(idx), warm_sigma=warm_sigma)
                raw_fits.append(fit)
                composite_pv += self._evaluate_pv(x, fit)
            except Exception:
                pass

        # Quality filter — amplitude threshold relative to spectrum's own intensity range
        min_amp = self.MIN_FIT_AMP_FRAC * float(np.nanmax(y)) if len(y) and np.nanmax(y) > 0 else 0.0
        kept = [
            f for f in raw_fits
            if f["fwhm"] >= self.MIN_FIT_FWHM
            and f["r2_local"] >= self.MIN_FIT_R2
            and f["amplitude"] >= min_amp
        ]
        kept.sort(key=lambda p: p["center"])

        # Composite display reconstruction (plotting only — not used for fit-quality metrics)
        composite_pv = np.zeros_like(y)
        for f in kept:
            composite_pv += self._evaluate_pv(x, f)
        dx = float(np.mean(np.diff(x))) if len(x) > 1 else 0.02
        sigma_bl = max(self.sigma_baseline_deg / dx, 5.0)
        baseline = gaussian_filter1d(y - composite_pv, sigma=sigma_bl)
        y_fit = baseline + composite_pv

        # Fit-quality metric: unweighted mean of per-peak local R².
        # All kept peaks have r2_local >= MIN_FIT_R2 by construction (poor fits
        # were filtered above). Unweighted mean: amplitude weighting was tested
        # and rejected — it hides a single bad weak peak in a sea of perfect
        # strong ones, defeating the "honest fit-quality metric" purpose.
        if kept:
            r2 = float(np.mean([f["r2_local"] for f in kept]))
        else:
            r2 = 0.0

        # AIC/BIC from full-spectrum residual against the display reconstruction.
        # k_params = 6 per kept peak (4 PV + 2 linear bg fit per ROI). These are
        # display-only summary stats — not for formal model selection across pipelines.
        # sigma² is clipped to a small epsilon so a noiseless perfect fit yields
        # a very-negative-but-finite AIC (rather than -inf), keeping the metric
        # numerically usable in tests and tables.
        n = max(len(y), 1)
        ss_res_display = float(np.sum((y - y_fit) ** 2))
        k_params = max(1, 6 * len(kept))
        sigma2 = max(ss_res_display / n, 1e-12)
        ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        aic = 2 * k_params - 2 * ll
        bic = k_params * np.log(n) - 2 * ll

        # Build all_peaks list with relative intensity
        max_amp = max((f["amplitude"] for f in kept), default=1.0)
        max_amp = max(max_amp, 1e-9)
        all_peaks = []
        for f in kept:
            all_peaks.append({
                "center": float(f["center"]),
                "fwhm": float(f["fwhm"]),
                "amplitude": float(f["amplitude"]),
                "eta": float(f["eta"]),
                "d_spacing": _d_spacing(f["center"], xrd.wavelength),
                "relative_intensity": round(float(f["amplitude"]) / max_amp * 100.0, 1),
            })

        # Dominant peak (largest amplitude)
        if kept:
            dom = max(kept, key=lambda f: f["amplitude"])
            dominant = {
                "center": float(dom["center"]),
                "fwhm": float(dom["fwhm"]),
                "amplitude": float(dom["amplitude"]),
                "eta": float(dom["eta"]),
            }
        else:
            dominant = {"center": float("nan"), "fwhm": float("nan"),
                        "amplitude": float("nan"), "eta": float("nan")}

        return FitResult(
            name=xrd.name,
            dominant_peak=dominant,
            all_peaks=all_peaks,
            r_squared=float(r2),
            aic=float(aic),
            bic=float(bic),
            n_peaks=len(kept),
            wavelength=xrd.wavelength,
            x=x,
            y=y,
            y_fit=y_fit,
            baseline=baseline,
            noise_std=float(noise),
            detected_peak_count=len(peak_indices),
        )

    def fit_batch(
        self, samples: dict[str, XRDData], progress: bool = True, warm_start: bool = False
    ) -> dict[str, FitResult]:
        results: dict[str, FitResult] = {}
        items = list(samples.items())
        it = tqdm(items, desc="Fitting", unit="sample") if progress else items
        warm_peaks: list | None = None
        for name, xrd in it:
            try:
                result = self.fit_sample(xrd, warm_peaks=warm_peaks if warm_start else None)
                results[name] = result
                if warm_start:
                    warm_peaks = result.all_peaks
            except Exception as exc:
                tqdm.write(f"  WARNING: {name} failed — {exc}")
        return results

    def _detect_peaks(self, x: np.ndarray, y: np.ndarray):
        dx = float(np.mean(np.diff(x))) if len(x) > 1 else 0.02
        sigma_signal = max(1.0, self.sigma_signal_deg / dx)
        sigma_baseline = max(sigma_signal * 5.0, self.sigma_baseline_deg / dx)

        y_signal = gaussian_filter1d(y, sigma=sigma_signal)
        baseline = gaussian_filter1d(y, sigma=sigma_baseline)
        bandpass = y_signal - baseline

        # Bandpass noise: MAD on lower half (peak-free regions)
        low_half = bandpass[bandpass <= np.quantile(bandpass, 0.5)]
        if len(low_half) >= 4:
            noise = float(np.median(np.abs(low_half - np.median(low_half))) / 0.6745)
        else:
            noise = float(np.std(bandpass)) * 0.5
        noise = max(noise, 1e-6)

        distance = max(1, int(self.MIN_SEPARATION_DEG / dx))
        min_width_pts = max(2, int(self.MIN_FWHM_DETECT_DEG / dx))
        peak_indices, _ = find_peaks(
            bandpass,
            prominence=self.k_sigma * noise,
            width=min_width_pts,
            distance=distance,
        )
        return np.asarray(peak_indices, dtype=int), baseline, noise

    def _fit_one_roi(self, x: np.ndarray, y: np.ndarray, peak_idx: int, warm_sigma: float | None = None) -> dict:
        dx = float(np.mean(np.diff(x))) if len(x) > 1 else 0.02
        half_pts = max(5, int(self.roi_window_deg / dx))
        lo = max(0, peak_idx - half_pts)
        hi = min(len(x), peak_idx + half_pts + 1)
        xc, yc = x[lo:hi], y[lo:hi]

        bg = LinearModel(prefix="bg_")
        pv = PseudoVoigtModel(prefix="p_")
        model = bg + pv
        params = model.make_params()
        params["bg_intercept"].set(value=float(np.percentile(yc, 10)))
        params["bg_slope"].set(value=0.0)
        c0 = float(x[peak_idx])
        params["p_center"].set(value=c0, min=c0 - 0.3, max=c0 + 0.3)
        params["p_amplitude"].set(value=float(yc.max() - yc.min()) * 0.4, min=0)
        sigma_init = float(np.clip(warm_sigma, 0.005, 0.5)) if warm_sigma is not None else 0.05
        params["p_sigma"].set(value=sigma_init, min=0.005, max=0.5)
        params["p_fraction"].set(value=0.2, min=0.0, max=1.0)

        result = model.fit(yc, params, x=xc, max_nfev=200)
        ss_tot = float(np.sum((yc - yc.mean()) ** 2))
        ss_res = float(np.sum(result.residual ** 2))
        r2_local = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "center": result.params["p_center"].value,
            "fwhm": result.params["p_fwhm"].value,
            "amplitude": result.params["p_amplitude"].value,
            "sigma": result.params["p_sigma"].value,
            "eta": result.params["p_fraction"].value,
            "r2_local": r2_local,
        }

    @staticmethod
    def _evaluate_pv(x: np.ndarray, fit: dict) -> np.ndarray:
        """PseudoVoigt evaluated at x for one peak's fitted params, via lmfit
        to match the fitting model exactly."""
        pv = PseudoVoigtModel(prefix="p_")
        params = pv.make_params()
        params["p_center"].set(value=fit["center"])
        params["p_amplitude"].set(value=fit["amplitude"])
        params["p_sigma"].set(value=fit["sigma"])
        params["p_fraction"].set(value=fit["eta"])
        return pv.eval(params=params, x=x)

    @staticmethod
    def _empty_result(xrd: XRDData, x: np.ndarray, y: np.ndarray) -> FitResult:
        return FitResult(
            name=xrd.name,
            dominant_peak={"center": float("nan"), "fwhm": float("nan"),
                           "amplitude": float("nan"), "eta": float("nan")},
            all_peaks=[],
            r_squared=0.0,
            aic=0.0,
            bic=0.0,
            n_peaks=0,
            wavelength=xrd.wavelength,
            x=x,
            y=y,
            y_fit=np.zeros_like(y),
            baseline=np.zeros_like(y),
        )


def _d_spacing(center_2theta_deg: float, wavelength_A: float) -> float:
    theta = np.deg2rad(center_2theta_deg / 2.0)
    sin_t = np.sin(theta)
    if sin_t <= 0:
        return float("nan")
    return wavelength_A / (2.0 * sin_t)
