from __future__ import annotations
import base64
import io
from datetime import date
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for embedded use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .loader import XRDData
from .fitter import FitResult


class HTMLReporter:
    def __init__(self, template_dir: str | Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )

    def render(
        self,
        summary_table: pd.DataFrame,
        fit_results: dict[str, FitResult],
        xrd_data: dict[str, XRDData],
        metadata: dict,
    ) -> str:
        template = self.env.get_template("memo_template.html")

        figures = {}
        for name, fr in fit_results.items():
            if name in xrd_data:
                figures[name] = self._build_sample_figure(name, xrd_data[name].df, fr)

        table_html = self._style_table(summary_table)

        n_flagged = int((summary_table["Flag"] != "").sum())
        mean_r2 = float(summary_table["R²"].mean())
        exec_summary = (
            f"Batch XRD analysis of {metadata.get('sample_count', len(summary_table))} samples completed. "
            f"Mean fit quality R²={mean_r2:.3f}. "
            f"{n_flagged} sample(s) flagged for review. "
            f"Scherrer crystallite sizes and d-spacings reported; "
            f"instrument broadening not subtracted (sizes are upper-bound estimates)."
        )

        best_idx = summary_table["R²"].idxmax()
        worst_idx = summary_table["R²"].idxmin()
        best_row = summary_table.loc[best_idx]
        worst_row = summary_table.loc[worst_idx]
        size_min = summary_table["Crystallite Size (nm)"].min()
        size_max = summary_table["Crystallite Size (nm)"].max()
        key_findings = [
            f"Best fit: {best_row['Sample']} (R²={best_row['R²']:.4f}).",
            (
                f"Poorest fit: {worst_row['Sample']} (R²={worst_row['R²']:.4f}). "
                + ("Manual inspection recommended." if worst_row["R²"] < 0.95 else "All fits acceptable.")
            ),
            (
                f"Crystallite sizes range from {size_min:.1f} nm to {size_max:.1f} nm "
                f"(Scherrer K=0.9, instrument broadening not subtracted)."
            ),
        ]

        flagged = summary_table[summary_table["Flag"] != ""]
        anomaly_report = flagged.to_dict(orient="records")

        return template.render(
            metadata=metadata,
            today=date.today().isoformat(),
            exec_summary=exec_summary,
            table_html=table_html,
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
        # Highlight poor fits
        if "R²" in df.columns:
            styler = styler.apply(
                lambda col: [
                    "background-color: #ffe0e0" if (isinstance(v, (int, float)) and v < 0.95) else ""
                    for v in col
                ],
                subset=["R²"],
            )
        return styler.to_html(table_uuid="xrd-summary-table")

    def _build_sample_figure(
        self, name: str, df: pd.DataFrame, fr: FitResult
    ) -> str:
        fig = plt.figure(figsize=(14, 3.5))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        x = df["two_theta"].values
        y = df["intensity"].values

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(x, y, color="#222", lw=0.8)
        ax1.set_title("Raw spectrum", fontsize=9)
        ax1.set_xlabel("2θ (°)", fontsize=8)
        ax1.set_ylabel("Intensity (norm.)", fontsize=8)
        ax1.tick_params(labelsize=7)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(x, y, color="#aaa", lw=0.8, label="Data", zorder=1)
        if fr.lmfit_result is not None:
            y_fit = fr.lmfit_result.best_fit
            ax2.plot(x, y_fit, color="#c0392b", lw=1.5, label="Fit", zorder=2)
        ax2.set_title(f"Fit overlay  R²={fr.r_squared:.4f}", fontsize=9)
        ax2.set_xlabel("2θ (°)", fontsize=8)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.tick_params(labelsize=7)

        ax3 = fig.add_subplot(gs[2])
        if fr.lmfit_result is not None:
            residuals = fr.lmfit_result.residual
            ax3.plot(x, residuals, color="#2980b9", lw=0.8)
            ax3.axhline(0, color="#888", lw=0.6, ls="--")
        ax3.set_title("Residuals", fontsize=9)
        ax3.set_xlabel("2θ (°)", fontsize=8)
        ax3.tick_params(labelsize=7)

        fig.suptitle(
            f"{name}  |  AIC={fr.aic:.1f}  BIC={fr.bic:.1f}  N_peaks={fr.n_peaks}",
            fontsize=9,
            y=1.02,
        )
        plt.tight_layout()
        b64 = self._fig_to_b64(fig)
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
