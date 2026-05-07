from __future__ import annotations
import numpy as np
import pandas as pd

from .fitter import FitResult


def _default_parse_phase(name: str) -> str:
    """Extract phase name from sample name by splitting on '__'."""
    return name.split("__")[0]


class XRDAnalyzer:
    K_SCHERRER = 0.9
    ANGSTROM_TO_NM = 0.1
    CRYSTALLITE_SIZE_CAP_NM = 200.0
    POOR_FIT_R2_THRESHOLD = 0.70
    # Calibrated for the fitter's unweighted mean of per-peak local R². Minor
    # reflections near the noise floor naturally have local R² in the 0.5-0.7
    # range, dragging the unweighted mean. Old 0.90 was for an inflated
    # self-referential composite R² that was removed.

    ZSCORE_THRESHOLD = 2.5

    @staticmethod
    def _scherrer(fwhm_deg: float, center_2theta_deg: float, wavelength_A: float) -> float:
        beta = np.deg2rad(fwhm_deg)
        theta = np.deg2rad(center_2theta_deg / 2.0)
        if beta <= 0 or np.cos(theta) == 0:
            return np.nan
        D_A = (XRDAnalyzer.K_SCHERRER * wavelength_A) / (beta * np.cos(theta))
        D_nm = D_A * XRDAnalyzer.ANGSTROM_TO_NM
        return min(D_nm, XRDAnalyzer.CRYSTALLITE_SIZE_CAP_NM)

    @staticmethod
    def _d_spacing(center_2theta_deg: float, wavelength_A: float) -> float:
        theta = np.deg2rad(center_2theta_deg / 2.0)
        sin_t = np.sin(theta)
        if sin_t <= 0:
            return np.nan
        return wavelength_A / (2.0 * sin_t)

    @classmethod
    def build_summary_table(cls, fit_results: dict[str, FitResult], parse_phase=None) -> pd.DataFrame:
        if parse_phase is None:
            parse_phase = _default_parse_phase
        rows = []
        for name, fr in fit_results.items():
            dp = fr.dominant_peak
            center = dp.get("center", np.nan)
            fwhm = dp.get("fwhm", np.nan)
            lam = fr.wavelength

            d_sp = cls._d_spacing(center, lam) if not np.isnan(center) else np.nan
            x_size = cls._scherrer(fwhm, center, lam) if not (np.isnan(fwhm) or np.isnan(center)) else np.nan

            phase = parse_phase(name)

            rows.append({
                "Sample": name,
                "Phase": phase,
                "2θ (°)": round(float(center), 3) if not np.isnan(center) else np.nan,
                "d-spacing (Å)": round(float(d_sp), 4) if not np.isnan(d_sp) else np.nan,
                "FWHM (°)": round(float(fwhm), 4) if not np.isnan(fwhm) else np.nan,
                "Crystallite Size (nm)": round(float(x_size), 1) if not np.isnan(x_size) else np.nan,
                "R²": round(float(fr.r_squared), 4),
                "AIC": round(float(fr.aic), 1),
                "BIC": round(float(fr.bic), 1),
                "N_peaks": int(fr.n_peaks),
                "Flag": "",
            })

        return pd.DataFrame(rows)

    @classmethod
    def flag_outliers(cls, table: pd.DataFrame) -> pd.DataFrame:
        df = table.copy()

        fwhm_vals = df["FWHM (°)"].fillna(df["FWHM (°)"].median())
        size_vals = df["Crystallite Size (nm)"].fillna(df["Crystallite Size (nm)"].median())

        def iqr_outlier(series: pd.Series, k: float = 1.5) -> pd.Series:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            return (series < q1 - k * iqr) | (series > q3 + k * iqr)

        fwhm_iqr = iqr_outlier(fwhm_vals)
        size_iqr = iqr_outlier(size_vals)

        flags = []
        for i in range(len(df)):
            parts = []
            r2_val = df.iloc[i]["R²"]
            if r2_val < cls.POOR_FIT_R2_THRESHOLD:
                parts.append(f"Poor fit (R²={r2_val:.3f})")
            size_val = df.iloc[i]["Crystallite Size (nm)"]
            if not np.isnan(size_val) and size_val >= cls.CRYSTALLITE_SIZE_CAP_NM:
                parts.append("Instrument-limited")
            if fwhm_iqr.iloc[i]:
                parts.append("FWHM outlier (IQR)")
            if size_iqr.iloc[i]:
                parts.append("Size outlier (IQR)")
            flags.append("; ".join(parts))

        df["Flag"] = flags
        return df

    @classmethod
    def build_peak_table(cls, fit_results: dict[str, FitResult], parse_phase=None) -> pd.DataFrame:
        """One row per fitted peak across all samples — position, FWHM, d-spacing, relative intensity."""
        if parse_phase is None:
            parse_phase = _default_parse_phase
        rows = []
        for name, fr in fit_results.items():
            phase = parse_phase(name)
            lam = fr.wavelength
            for peak_num, pk in enumerate(fr.all_peaks, start=1):
                center = pk["center"]
                fwhm = pk["fwhm"]
                x_size = cls._scherrer(fwhm, center, lam) if fwhm > 0 else np.nan
                rows.append({
                    "Sample": name,
                    "Phase": phase,
                    "Peak #": peak_num,
                    "2θ (°)": round(center, 3),
                    "d-spacing (Å)": round(pk["d_spacing"], 4) if not np.isnan(pk["d_spacing"]) else np.nan,
                    "FWHM (°)": round(fwhm, 4),
                    "Crystallite Size (nm)": round(x_size, 1) if not np.isnan(x_size) else np.nan,
                    "Rel. Intensity (%)": pk["relative_intensity"],
                    "η": round(pk["eta"], 3),
                })
        return pd.DataFrame(rows)
