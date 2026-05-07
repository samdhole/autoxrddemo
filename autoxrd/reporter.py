from __future__ import annotations
import base64
import io
from datetime import date
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for embedded use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .loader import XRDData
from .fitter import FitResult


class HTMLReporter:
    def __init__(self, template_dir: str | Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
        )

    def render(
        self,
        summary_table: pd.DataFrame,
        fit_results: dict[str, FitResult],
        xrd_data: dict[str, XRDData],
        metadata: dict,
        peak_table: pd.DataFrame | None = None,
    ) -> str:
        template = self.env.get_template("memo_template.html")

        figures = {}
        for name, fr in fit_results.items():
            if name in xrd_data:
                figures[name] = self._build_sample_figure(name, xrd_data[name].df, fr)

        table_html = self._style_table(summary_table)

        # Guard up-front: empty table or schema mismatch should not crash the renderer.
        has_flag = "Flag" in summary_table.columns
        has_r2 = "R²" in summary_table.columns
        is_empty = len(summary_table) == 0

        if is_empty or not (has_flag and has_r2):
            n_flagged = 0
            mean_r2 = float("nan")
        else:
            n_flagged = int((summary_table["Flag"] != "").sum())
            mean_r2 = float(summary_table["R²"].mean())

        exec_summary = (
            f"Batch XRD analysis of {metadata.get('sample_count', len(summary_table))} samples completed. "
            f"Mean fit quality R²={mean_r2:.3f}. "
            f"{n_flagged} sample(s) flagged for review. "
            f"Scherrer crystallite sizes and d-spacings reported; "
            f"instrument broadening not subtracted (sizes are upper-bound estimates)."
        )

        if is_empty or not has_r2:
            key_findings = ["No samples in summary table."]
        else:
            r2_series = summary_table["R²"].dropna()
            size_series = summary_table["Crystallite Size (nm)"].dropna()
            if r2_series.empty:
                key_findings = ["No valid R² values in summary table."]
            else:
                best_row = summary_table.loc[r2_series.idxmax()]
                worst_row = summary_table.loc[r2_series.idxmin()]
                from .analyzer import XRDAnalyzer
                threshold = XRDAnalyzer.POOR_FIT_R2_THRESHOLD
                if len(summary_table) == 1:
                    fit_line = f"Single sample: {best_row['Sample']} (R²={best_row['R²']:.4f})."
                else:
                    fit_line = (
                        f"Poorest fit: {worst_row['Sample']} (R²={worst_row['R²']:.4f}). "
                        + ("Manual inspection recommended." if worst_row["R²"] < threshold else "All fits acceptable.")
                    )
                if size_series.empty:
                    size_line = "Crystallite size unavailable (no valid Scherrer estimates)."
                else:
                    size_line = (
                        f"Crystallite sizes range from {size_series.min():.1f} nm to {size_series.max():.1f} nm "
                        f"(Scherrer K=0.9, instrument broadening not subtracted)."
                    )
                key_findings = [
                    f"Best fit: {best_row['Sample']} (R²={best_row['R²']:.4f}).",
                    fit_line,
                    size_line,
                ]

        if has_flag and not is_empty:
            flagged = summary_table[summary_table["Flag"] != ""]
            anomaly_report = flagged.to_dict(orient="records")
        else:
            anomaly_report = []

        peak_table_html = self._style_peak_table(peak_table) if peak_table is not None and not peak_table.empty else None

        return template.render(
            metadata=metadata,
            today=date.today().isoformat(),
            exec_summary=exec_summary,
            table_html=table_html,
            peak_table_html=peak_table_html,
            figures=figures,
            summary_table=summary_table.to_dict(orient="records"),
            anomaly_report=anomaly_report,
            key_findings=key_findings,
        )

    def _style_table(self, df: pd.DataFrame) -> str:
        fmt = {
            "2θ (°)": "{:.3f}",
            "d-spacing (Å)": "{:.4f}",
            "FWHM (°)": "{:.4f}",
            "Crystallite Size (nm)": "{:.1f}",
            "R²": "{:.4f}",
            "AIC": "{:.1f}",
            "BIC": "{:.1f}",
        }
        # Only format columns that exist in the dataframe
        fmt = {k: v for k, v in fmt.items() if k in df.columns}

        styler = df.style.format(fmt, na_rep="—")
        # Highlight poor fits — threshold tracks XRDAnalyzer.POOR_FIT_R2_THRESHOLD
        # so the cell color and the Flag column agree.
        if "R²" in df.columns:
            from .analyzer import XRDAnalyzer
            threshold = XRDAnalyzer.POOR_FIT_R2_THRESHOLD
            styler = styler.apply(
                lambda col: [
                    "background-color: #ffc0c0" if (isinstance(v, (int, float)) and v < threshold) else
                    ""
                    for v in col
                ],
                subset=["R²"],
            )
        return styler.to_html(table_uuid="xrd-summary-table")

    def _build_sample_figure(
        self, name: str, df: pd.DataFrame, fr: FitResult
    ) -> str:
        fig = plt.figure(figsize=(14, 3.5), constrained_layout=True)

        x = df["two_theta"].values
        y = df["intensity"].values

        peak_centers = [p["center"] for p in fr.all_peaks]

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(x, y, color="#222", lw=0.8)
        for pc in peak_centers:
            ax1.axvline(pc, color="#c0392b", lw=0.6, alpha=0.5, ls="--")
        ax1.set_title("Raw spectrum", fontsize=9)
        ax1.set_xlabel("2θ (°)", fontsize=8)
        ax1.set_ylabel("Intensity (norm.)", fontsize=8)
        ax1.tick_params(labelsize=7)

        x_fit = fr.x if fr.x.size else x
        y_data = fr.y if fr.y.size else y
        y_fit = fr.y_fit if fr.y_fit.size else None
        baseline = fr.baseline if fr.baseline.size else None

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(x_fit, y_data, color="#aaa", lw=0.8, label="Data", zorder=1)
        if y_fit is not None:
            ax2.plot(x_fit, y_fit, color="#c0392b", lw=1.5, label="Fit", zorder=2)
        if baseline is not None:
            ax2.plot(x_fit, baseline, color="#7f8c8d", lw=0.6, ls=":", label="Baseline", zorder=1)
        for j, pc in enumerate(peak_centers, start=1):
            ax2.axvline(pc, color="#2980b9", lw=0.6, alpha=0.6, ls="--")
            ax2.text(pc, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] else 95,
                     str(j), fontsize=5, ha="center", color="#2980b9", va="top")
        ax2.set_title(f"Fit overlay  R²={fr.r_squared:.4f}", fontsize=9)
        ax2.set_xlabel("2θ (°)", fontsize=8)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.tick_params(labelsize=7)

        ax3 = fig.add_subplot(1, 3, 3)
        if y_fit is not None:
            residual = y_data - y_fit
            ax3.plot(x_fit, residual, color="#2980b9", lw=0.8)
            ax3.axhline(0, color="#888", lw=0.6, ls="--")
        ax3.set_title("Residuals", fontsize=9)
        ax3.set_xlabel("2θ (°)", fontsize=8)
        ax3.tick_params(labelsize=7)

        fig.suptitle(
            f"{name}  |  AIC={fr.aic:.1f}  BIC={fr.bic:.1f}  N_peaks={fr.n_peaks}",
            fontsize=9,
        )
        b64 = self._fig_to_b64(fig)
        plt.close(fig)
        return b64

    def _style_peak_table(self, df: pd.DataFrame) -> str:
        fmt = {
            "2θ (°)": "{:.3f}",
            "d-spacing (Å)": "{:.4f}",
            "FWHM (°)": "{:.4f}",
            "Crystallite Size (nm)": "{:.1f}",
            "Rel. Intensity (%)": "{:.1f}",
            "η": "{:.3f}",
        }
        fmt = {k: v for k, v in fmt.items() if k in df.columns}
        return df.style.format(fmt, na_rep="—").to_html(table_uuid="xrd-peak-table")

    @staticmethod
    def build_trend_figure(
        peak_table: pd.DataFrame,
        sample_order: list[str],
        trend_models: pd.DataFrame | None = None,
        parse_phase=None,
    ) -> str:
        """Line plots of peak position and FWHM vs. sample index for same-material batches.

        If trend_models (output of XRDAnalyzer.build_trend_model) is provided, overlays
        dashed linear fit lines on each subplot in matching colors.
        """
        if parse_phase is None:
            parse_phase = lambda s: s.split("__")[0] if "__" in s else s

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        sample_idx = {name: i + 1 for i, name in enumerate(sample_order)}
        peak_table = peak_table[peak_table["Sample"].isin(sample_order)].copy()
        peak_table["_idx"] = peak_table["Sample"].map(sample_idx)

        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
        n_samples = len(sample_order)
        x_line = np.array([1.0, float(n_samples)])

        for j, (peak_num, grp) in enumerate(peak_table.groupby("Peak #")):
            grp = grp.sort_values("_idx")
            label = f"Peak {peak_num} (~{grp['2θ (°)'].mean():.1f}°)"
            c = colors[j % len(colors)]
            ax1.plot(grp["_idx"], grp["2θ (°)"], marker="o", ms=5, lw=1.4, color=c, label=label)
            ax2.plot(grp["_idx"], grp["FWHM (°)"], marker="o", ms=5, lw=1.4, color=c, label=label)

            if trend_models is not None and not trend_models.empty:
                model_row = trend_models[trend_models["Peak #"] == peak_num]
                if not model_row.empty:
                    row = model_row.iloc[0]
                    pos_slope = row["Position slope (°/sample)"]
                    fwhm_slope = row["FWHM slope (°/sample)"]
                    mean_obs_idx = float(grp["_idx"].mean())
                    if not np.isnan(float(pos_slope)):
                        mean_pos = float(grp["2θ (°)"].mean())
                        intercept_pos = mean_pos - pos_slope * mean_obs_idx
                        ax1.plot(x_line, pos_slope * x_line + intercept_pos,
                                 ls="--", lw=1.0, color=c, alpha=0.55)
                    if not np.isnan(float(fwhm_slope)):
                        mean_fwhm = float(grp["FWHM (°)"].mean())
                        intercept_fwhm = mean_fwhm - fwhm_slope * mean_obs_idx
                        ax2.plot(x_line, fwhm_slope * x_line + intercept_fwhm,
                                 ls="--", lw=1.0, color=c, alpha=0.55)

        tick_labels = [parse_phase(s) for s in sample_order]

        ax1.set_title("Peak Position vs. Sample", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Sample index", fontsize=9)
        ax1.set_ylabel("2θ (°)", fontsize=9)
        ax1.set_xticks(list(sample_idx.values()))
        ax1.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)
        ax1.legend(fontsize=7, loc="best")

        ax2.set_title("FWHM vs. Sample", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Sample index", fontsize=9)
        ax2.set_ylabel("FWHM (°)", fontsize=9)
        ax2.set_xticks(list(sample_idx.values()))
        ax2.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)

        fig.suptitle("Peak Tracking Across Batch", fontsize=11, fontweight="bold")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return b64

    @staticmethod
    def _fig_to_b64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return encoded
