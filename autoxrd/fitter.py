from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm
from lmfit.models import PseudoVoigtModel, ConstantModel

from .loader import XRDData


@dataclass
class FitResult:
    name: str
    lmfit_result: object          # lmfit ModelResult
    dominant_peak: dict           # center, fwhm, amplitude, eta
    r_squared: float
    aic: float
    bic: float
    n_peaks: int
    wavelength: float


class XRDFitter:
    MAX_PEAKS = 15

    def __init__(self, prominence: float = 8.0, min_distance_deg: float = 0.5):
        self.prominence = prominence
        self.min_distance_deg = min_distance_deg

    def fit_sample(self, xrd: XRDData) -> FitResult:
        x = xrd.df["two_theta"].values
        y = xrd.df["intensity"].values

        dx = np.mean(np.diff(x)) if len(x) > 1 else 0.02
        min_dist_pts = max(1, int(self.min_distance_deg / dx))

        peak_indices, props = find_peaks(
            y, prominence=self.prominence, distance=min_dist_pts
        )

        if len(peak_indices) == 0:
            # Retry at half prominence (catches broad low-intensity peaks like Kaolinite).
            peak_indices, props = find_peaks(
                y, prominence=self.prominence / 2, distance=min_dist_pts
            )
        if len(peak_indices) == 0:
            # Last resort: global max. props stays empty so the cap block
            # below uses raw intensity as a proxy — safe for one peak.
            peak_indices = np.array([int(np.argmax(y))])
            props = {}

        # Keep top MAX_PEAKS by prominence
        if len(peak_indices) > self.MAX_PEAKS:
            prominences = props.get("prominences", y[peak_indices])
            top = np.argsort(prominences)[::-1][: self.MAX_PEAKS]
            peak_indices = peak_indices[top]

        model = ConstantModel(prefix="bg_")
        params = model.make_params(bg_c=float(np.percentile(y, 5)))

        for i, idx in enumerate(peak_indices):
            pv = PseudoVoigtModel(prefix=f"p{i}_")
            pv_params = pv.make_params()
            c = float(x[idx])
            pv_params[f"p{i}_center"].set(value=c, min=c - 1.5, max=c + 1.5)
            pv_params[f"p{i}_amplitude"].set(value=float(y[idx]) * 0.5, min=0)
            pv_params[f"p{i}_sigma"].set(value=0.1, min=0.005, max=3.0)
            pv_params[f"p{i}_fraction"].set(value=0.5, min=0.0, max=1.0)
            params.update(pv_params)
            model = model + pv

        result = model.fit(y, params, x=x)

        ss_res = np.sum(result.residual ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        dominant = self._dominant_peak(result, len(peak_indices))

        return FitResult(
            name=xrd.name,
            lmfit_result=result,
            dominant_peak=dominant,
            r_squared=r2,
            aic=float(result.aic),
            bic=float(result.bic),
            n_peaks=len(peak_indices),
            wavelength=xrd.wavelength,
        )

    def fit_batch(
        self, samples: dict[str, XRDData], progress: bool = True
    ) -> dict[str, FitResult]:
        # Failed samples are excluded from results entirely (not stored as None).
        # Callers should compare len(results) vs len(samples) if completeness matters.
        results: dict[str, FitResult] = {}
        items = list(samples.items())
        it = tqdm(items, desc="Fitting", unit="sample") if progress else items
        for name, xrd in it:
            try:
                results[name] = self.fit_sample(xrd)
            except Exception as exc:
                tqdm.write(f"  WARNING: {name} failed — {exc}")
        return results

    @staticmethod
    def _dominant_peak(result, n_peaks: int) -> dict:
        best_amp = -np.inf
        best: dict = {}
        for i in range(n_peaks):
            amp_key = f"p{i}_amplitude"
            if amp_key not in result.params:
                continue
            amp_val = result.params[amp_key].value
            fwhm_key = f"p{i}_fwhm"
            if fwhm_key not in result.params:
                continue
            if amp_val > best_amp:
                best_amp = amp_val
                best = {
                    "center": result.params[f"p{i}_center"].value,
                    "fwhm": result.params[fwhm_key].value,
                    "amplitude": amp_val,
                    "eta": result.params[f"p{i}_fraction"].value,
                }
        return best
