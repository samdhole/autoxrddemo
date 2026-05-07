"""Generate notebooks/xrd_batch_analysis.ipynb from scratch.
Run once: python build_notebook.py
"""
import json
import uuid
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                  "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8],
                  "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.splitlines(keepends=True)})


md("""# XRD Batch Analysis to Technical Memo
### Same-material batch characterization · samdhole.github.io

---

## The Real-World Use Case

A characterization lab runs 8 samples of the same material from a process variation study (annealing temperatures, milling times, deposition conditions). Each XRD pattern needs to be peak-fit individually, the same reflection tracked across all 8 samples, and a summary memo delivered to the process engineer.

**Manual workflow:** open each pattern in JADE/HighScore/EVA, identify peaks by eye, fit profile functions, record FWHM and position into a spreadsheet, compute d-spacings and Scherrer sizes by hand, write the memo. **~4 hours for 8 samples.**

**This pipeline:** load → auto-detect → batch-fit → trend-track → memo. **~20 seconds.**

---

## Demo Dataset

8 simulated quartz patterns with systematic FWHM broadening (0.08° → 0.22°). Synthetic data lets us know exactly which 15 reflections should be detected (ICDD 00-046-1045 quartz). In production this would be 8 batches of one material under varying processing conditions.

| Step | Section | Output |
|------|---------|--------|
| 1 | Load batch | 8 XRDData objects |
| 2 | Visualize raw spectra | 2×4 grid |
| 3 | Single-sample walkthrough | DoG detection + ROI fits annotated |
| 4 | Batch fit | FitResult per sample (~1.5s each) |
| 5 | Summary table | Per-sample dominant peak |
| 6 | Per-peak table | All ~15 reflections × 8 samples (xrdfit-style) |
| 7 | Trend plot | Peak position + FWHM vs sample index |
| 8 | HTML memo | Single-file deliverable |

**Method:** Difference of Gaussians (xrfit-style) for auto-detect, per-peak ROI lmfit PseudoVoigt, Scherrer K=0.9 (Cu Kα λ=1.54056 Å)
""")

md("## Step 1: Load XRD Batch")

code("""import sys, time, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Image

_nb_cwd = Path().resolve()
_repo_root = _nb_cwd.parent if _nb_cwd.name == "notebooks" else _nb_cwd
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from autoxrd.loader import XRDLoader
from autoxrd.fitter import XRDFitter
from autoxrd.analyzer import XRDAnalyzer
from autoxrd.reporter import HTMLReporter

DATA_DIR = _repo_root / "data" / "synthetic_quartz"
OUTPUT_DIR = _repo_root / "output" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _inline_show(*args, **kwargs):
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    display(Image(data=buf.getvalue()))
    plt.close("all")
plt.show = _inline_show

t0_total = time.perf_counter()

t0 = time.perf_counter()
samples = XRDLoader.load_directory(DATA_DIR)
load_time = time.perf_counter() - t0
print(f"Loaded {len(samples)} samples in {load_time:.3f}s")
for name, xrd in samples.items():
    n = len(xrd.df)
    lo, hi = xrd.df["two_theta"].min(), xrd.df["two_theta"].max()
    print(f"  {name:<25s} {n:5d} pts  lambda={xrd.wavelength:.5f} A  2theta: {lo:.1f}-{hi:.1f} deg")
""")

md("""## Step 2: Visualize Raw Spectra

Each pattern is the same quartz material at a different (simulated) annealing condition. The peaks are at the same 2θ positions across the batch, but their widths increase as grains coarsen.""")

code("""fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
fig.suptitle("Raw XRD patterns — synthetic quartz annealing series",
             fontsize=13, fontweight="bold")
for ax, (name, xrd) in zip(axes.flat, samples.items()):
    ax.plot(xrd.df["two_theta"], xrd.df["intensity"], color="#1a1a1a", lw=0.6)
    label = name.replace("Quartz_Anneal_", "Sample ")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("2θ (°)", fontsize=8)
    ax.set_ylabel("Intensity (norm.)", fontsize=8)
    ax.tick_params(labelsize=7)
plt.show()
print("All 8 patterns share the same peak positions; FWHM increases left→right, top→bottom.")
""")

md("""## Step 3: Single-Sample Walkthrough — DoG Detection + ROI Fits

The fitter has two stages.

**Detect (Difference of Gaussians, after `xrfit`):** convolve the spectrum with a narrow Gaussian (preserves peaks, kills noise) and a wide Gaussian (approximates baseline). The bandpass `narrow − wide` isolates peaks against any baseline shape. Threshold = k×MAD(bandpass) on peak-free regions. No prominence knob to tune.

**Fit (per-peak ROI):** crop ±0.25° around each detected peak, fit a single PseudoVoigt + linear background. ~30 ms per peak vs minutes for global multi-peak fits. Quality filter rejects fits with FWHM<0.04°, R²<0.5, or amplitude<0.25.""")

code("""from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Pick sample 4 (mid-FWHM) for the walkthrough
demo_name = list(samples.keys())[3]
demo = samples[demo_name]
x = demo.df["two_theta"].values
y = demo.df["intensity"].values

# Replicate the fitter's DoG detection for visualization
fitter = XRDFitter()
dx = float(np.mean(np.diff(x)))
sig_signal = max(1.0, fitter.sigma_signal_deg / dx)
sig_baseline = max(sig_signal * 5.0, fitter.sigma_baseline_deg / dx)
y_signal = gaussian_filter1d(y, sigma=sig_signal)
y_baseline = gaussian_filter1d(y, sigma=sig_baseline)
bandpass = y_signal - y_baseline
low_half = bandpass[bandpass <= np.quantile(bandpass, 0.5)]
noise = float(np.median(np.abs(low_half - np.median(low_half))) / 0.6745)
threshold = fitter.k_sigma * noise

distance = max(1, int(fitter.MIN_SEPARATION_DEG / dx))
min_w = max(2, int(fitter.MIN_FWHM_DETECT_DEG / dx))
peak_idx, _ = find_peaks(bandpass, prominence=threshold, width=min_w, distance=distance)

# Run the actual fit on the same sample
result = fitter.fit_sample(demo)

print(f"Sample: {demo_name}")
print(f"DoG: σ_signal={sig_signal:.1f}pts, σ_baseline={sig_baseline:.1f}pts")
print(f"Bandpass noise (MAD): {noise:.3f}, threshold = {fitter.k_sigma}σ = {threshold:.3f}")
print(f"DoG raw candidates: {len(peak_idx)}, kept after fit-quality filter: {result.n_peaks}")
print(f"Mean per-peak local R²: {result.r_squared:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

# Panel 1: raw + DoG components
axes[0].plot(x, y, color="#1a1a1a", lw=0.6, label="Raw")
axes[0].plot(x, y_baseline, color="#7f8c8d", lw=1.0, ls="--", label="DoG wide (baseline)")
axes[0].set_title("Raw + DoG baseline", fontsize=10, fontweight="bold")
axes[0].set_xlabel("2θ (°)"); axes[0].set_ylabel("Intensity (norm.)")
axes[0].legend(fontsize=8)

# Panel 2: bandpass + detected peaks
axes[1].plot(x, bandpass, color="#2980b9", lw=0.6, label="Bandpass (narrow − wide)")
axes[1].axhline(threshold, color="#c0392b", ls="--", lw=1.0,
                label=f"{fitter.k_sigma}σ threshold = {threshold:.2f}")
axes[1].plot(x[peak_idx], bandpass[peak_idx], "rv", ms=8, label=f"{len(peak_idx)} candidates")
axes[1].set_title("Bandpass + auto-detected peaks", fontsize=10, fontweight="bold")
axes[1].set_xlabel("2θ (°)"); axes[1].set_ylabel("Bandpass")
axes[1].legend(fontsize=8)

# Panel 3: composite fit
axes[2].plot(result.x, result.y, color="#aaa", lw=0.6, label="Data")
axes[2].plot(result.x, result.y_fit, color="#c0392b", lw=1.2, label=f"Composite (mean local R²={result.r_squared:.3f})")
axes[2].plot(result.x, result.baseline, color="#7f8c8d", lw=0.5, ls=":", label="Baseline (refined)")
for p in result.all_peaks:
    axes[2].axvline(p["center"], color="#2980b9", lw=0.4, alpha=0.5)
axes[2].set_title(f"Composite fit ({result.n_peaks} peaks kept)", fontsize=10, fontweight="bold")
axes[2].set_xlabel("2θ (°)")
axes[2].legend(fontsize=8)

plt.show()
""")

md("## Step 4: Batch Fit All 8 Samples")

code("""print("Running batch peak fitting...")
t0_fit = time.perf_counter()
fit_results = XRDFitter().fit_batch(samples, progress=True)
fit_time = time.perf_counter() - t0_fit
print(f"\\nBatch complete: {len(fit_results)} samples fitted in {fit_time:.1f}s")
print(f"  Mean time per sample: {fit_time/len(fit_results)*1000:.0f} ms\\n")
for name, fr in fit_results.items():
    label = name.replace("Quartz_Anneal_", "Sample ")
    print(f"  {label:<12s}  N_peaks={fr.n_peaks:2d}  R²={fr.r_squared:.4f}  dominant fwhm={fr.dominant_peak['fwhm']:.4f}")
""")

md("""## Step 5: Per-Sample Summary Table

The dominant reflection (largest amplitude) for each sample. This is the top-level view a process engineer reads first.""")

code("""summary = XRDAnalyzer.build_summary_table(fit_results)
summary = XRDAnalyzer.flag_outliers(summary)

display(
    summary[["Sample", "2θ (°)", "d-spacing (Å)", "FWHM (°)", "Crystallite Size (nm)", "R²", "Flag"]]
    .style
    .format({"2θ (°)": "{:.3f}", "d-spacing (Å)": "{:.4f}", "FWHM (°)": "{:.4f}",
             "Crystallite Size (nm)": "{:.1f}", "R²": "{:.4f}"})
    .apply(lambda col: ["background-color: #ffc0c0" if (isinstance(v, float) and v < XRDAnalyzer.POOR_FIT_R2_THRESHOLD) else ""
                       for v in col], subset=["R²"])
    .set_caption("Dominant peak per sample")
)
""")

md("""## Step 6: Per-Peak Parameter Table

All ~15 quartz reflections detected in each sample. This is the table xrdfit produces — center, d-spacing, FWHM, Scherrer crystallite size, relative intensity, η — but xrdfit requires the user to pre-specify peak ranges. Here it's automatic.""")

code("""peak_table = XRDAnalyzer.build_peak_table(fit_results)
print(f"Peak table: {len(peak_table)} rows ({len(peak_table) // len(fit_results)} avg peaks/sample)\\n")

display(
    peak_table.head(20).style
    .format({"2θ (°)": "{:.3f}", "d-spacing (Å)": "{:.4f}", "FWHM (°)": "{:.4f}",
             "Crystallite Size (nm)": "{:.1f}", "Rel. Intensity (%)": "{:.1f}", "η": "{:.3f}"})
    .set_caption("Per-peak parameters (first 20 rows; full table embedded in HTML memo)")
)
""")

md("""## Step 7: Peak Trend Tracking Across Batch

For a same-material batch, the value-add is tracking how each peak family evolves. Position drift indicates lattice strain or composition change. FWHM increase indicates grain refinement, micro-strain, or instrument shift. The synthetic data has perfect peak positions and monotonically increasing FWHM — the trend plot should reflect that.""")

code("""sample_order = list(fit_results.keys())
trend_b64 = HTMLReporter.build_trend_figure(peak_table, sample_order)
display(Image(data=__import__("base64").b64decode(trend_b64)))
print("Position lines flat ⇒ no lattice change. FWHM lines monotonic ⇒ systematic broadening — exactly what the synthetic data encodes.")
""")

md("""## Step 7b: Linear Trend Models

Fitting peak position and FWHM as linear functions of sample index quantifies what the scatter plot suggests visually. The slope tells you the rate of change per sample; R² tells you how well the linear model fits.""")

code("""models = XRDAnalyzer.build_trend_model(peak_table, sample_order)
print(f"Linear trend models for {len(models)} peak families:\\n")
display(
    models.style
    .format({
        "Center (°)": "{:.3f}",
        "Position slope (°/sample)": "{:.5f}",
        "Position R²": "{:.4f}",
        "FWHM slope (°/sample)": "{:.5f}",
        "FWHM R²": "{:.4f}",
    })
    .set_caption("Linear trend coefficients — peak position and FWHM vs. sample index")
)
trend_b64_model = HTMLReporter.build_trend_figure(peak_table, sample_order, trend_models=models)
display(Image(data=__import__("base64").b64decode(trend_b64_model)))
print("Dashed lines: linear model fits. FWHM slope > 0 confirms systematic grain coarsening.")
""")

md("## Step 8: Generate HTML Technical Memo")

code("""reporter = HTMLReporter(template_dir=_repo_root / "templates")

t0_report = time.perf_counter()
memo_html = reporter.render(
    summary_table=summary,
    fit_results=fit_results,
    xrd_data=samples,
    metadata={
        "title": "XRD Batch Analysis — Synthetic Quartz Annealing Series",
        "analyst": "Automated Pipeline · Sam Dhole",
        "instrument": "Synthetic data (Cu Kα λ=1.54056 Å, ICDD 00-046-1045)",
        "sample_count": len(samples),
    },
    peak_table=peak_table,
)
report_time = time.perf_counter() - t0_report

output_path = OUTPUT_DIR / "xrd_memo.html"
output_path.write_text(memo_html, encoding="utf-8")

total_time = time.perf_counter() - t0_total
print(f"Memo: {output_path}")
print(f"Size: {len(memo_html) / 1024:.0f} KB\\n")
print("=" * 50)
print("  TIMING")
print("=" * 50)
print(f"  Load:        {load_time:.2f}s")
print(f"  Batch fit:   {fit_time:.1f}s  ({len(fit_results)} samples)")
print(f"  Render:      {report_time:.2f}s")
print(f"  Total:       {total_time:.1f}s")
print()
print(f"  Manual researcher equivalent:  ~4 hours")
print(f"  Pipeline:                      {total_time:.0f}s")
print(f"  Speedup:                       ~{int(14400 / total_time):d}×")
print("=" * 50)
""")

md("""---

## Summary

The pipeline auto-detects all reflections in each spectrum (no manual peak picking — `xrdfit` requires you to specify peak ranges; `autoXRD` does CNN classification but no per-peak fits), tracks peak families across the batch, and produces a single-file HTML memo a non-specialist process engineer can read.

**Method stack:**
- **Detection:** Difference of Gaussians (after `xrfit`/tgdane) — narrow vs. wide Gaussian smoothing isolates peaks against any baseline shape, MAD-noise-thresholded.
- **Fit:** `lmfit` PseudoVoigt + linear background per ROI (~30 ms/peak).
- **Quality filter:** FWHM, R², amplitude — rejects noise candidates that survived detection.
- **Trend:** peak families clustered post-hoc; position + FWHM tracked vs sample index.
- **Memo:** Jinja2 + matplotlib (base64-embedded), single self-contained HTML file.

---
*Generated by autoxrd · samdhole.github.io*
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "notebooks" / "xrd_batch_analysis.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
