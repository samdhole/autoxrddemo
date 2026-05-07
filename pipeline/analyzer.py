from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from .fitter import FitResult


class XRDAnalyzer:
    K_SCHERRER = 0.9
    ANGSTROM_TO_NM = 0.1
    CRYSTALLITE_SIZE_CAP_NM = 500.0
    POOR_FIT_R2_THRESHOLD = 0.95
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
    def build_summary_table(cls, fit_results: dict[str, FitResult]) -> pd.DataFrame:
        rows = []
        for name, fr in fit_results.items():
            dp = fr.dominant_peak
            center = dp.get("center", np.nan)
            fwhm = dp.get("fwhm", np.nan)
            lam = fr.wavelength

            d_sp = cls._d_spacing(center, lam) if not np.isnan(center) else np.nan
            x_size = cls._scherrer(fwhm, center, lam) if not (np.isnan(fwhm) or np.isnan(center)) else np.nan

            phase = name.split("__")[0]

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

        # Z-score flags (spec requirement)
        if len(fwhm_vals) > 1:
            fwhm_z = stats.zscore(fwhm_vals)
            size_z = stats.zscore(size_vals)
        else:
            fwhm_z = np.zeros(len(fwhm_vals))
            size_z = np.zeros(len(size_vals))

        # IQR flags (supplemental — more reliable at n=8)
        def iqr_outlier(series: pd.Series, k: float = 1.5) -> pd.Series:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            return (series < q1 - k * iqr) | (series > q3 + k * iqr)

        fwhm_iqr = iqr_outlier(fwhm_vals)
        size_iqr = iqr_outlier(size_vals)

        z_flags, iqr_flags = [], []
        for i in range(len(df)):
            parts = []
            if df.iloc[i]["R²"] < cls.POOR_FIT_R2_THRESHOLD:
                parts.append("Poor fit")
            if abs(fwhm_z[i]) > cls.ZSCORE_THRESHOLD:
                parts.append("Broad peak")
            if abs(size_z[i]) > cls.ZSCORE_THRESHOLD:
                parts.append("Anomalous grain size")
            z_flags.append("; ".join(parts))

            iqr_parts = []
            if fwhm_iqr.iloc[i]:
                iqr_parts.append("FWHM outlier (IQR)")
            if size_iqr.iloc[i]:
                iqr_parts.append("Size outlier (IQR)")
            iqr_flags.append("; ".join(iqr_parts))

        df["Flag"] = z_flags
        df["Flag (IQR supplement)"] = iqr_flags
        return df
