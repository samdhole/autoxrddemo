"""python -m pipeline  —  run the full XRD batch analysis pipeline."""
from __future__ import annotations
import sys
import time
from pathlib import Path

from .loader import XRDLoader
from .fitter import XRDFitter
from .analyzer import XRDAnalyzer
from .reporter import HTMLReporter

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic_quartz"
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "reports"
OUTPUT_FILE = OUTPUT_DIR / "xrd_memo.html"


def parse_phase(name: str) -> str:
    if name.startswith("Quartz_Anneal_"):
        return "Quartz"
    return name.split("__")[0]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # not available in all environments (e.g. IDLE)

    t0 = time.perf_counter()

    print("Loading samples...")
    samples = XRDLoader.load_directory(DATA_DIR)
    if not samples:
        raise RuntimeError(f"No .txt files found in {DATA_DIR} — run data/xrd_samples/download_samples.py first")
    print(f"  Loaded {len(samples)} samples")

    print("Batch fitting...")
    t1 = time.perf_counter()
    fitter = XRDFitter()
    fit_results = fitter.fit_batch(samples)
    t2 = time.perf_counter()
    print(f"  Fitted {len(fit_results)} samples in {t2 - t1:.1f}s")

    print("Building summary...")
    summary = XRDAnalyzer.build_summary_table(fit_results, parse_phase=parse_phase)
    summary = XRDAnalyzer.flag_outliers(summary)
    peak_table = XRDAnalyzer.build_peak_table(fit_results, parse_phase=parse_phase)

    print("Rendering HTML memo...")
    reporter = HTMLReporter(TEMPLATE_DIR)
    html = reporter.render(
        summary_table=summary,
        fit_results=fit_results,
        xrd_data=samples,
        metadata={
            "title": "XRD Batch Analysis — Decision Memo",
            "analyst": "Automated Pipeline · Sam Dhole",
            "instrument": "RRUFF database · wavelength per file header (default Cu Kα λ=1.54056 Å)",
            "sample_count": len(samples),
            "portfolio_context": (
                "This portfolio demo turns repeat materials-characterization data into a decision-ready technical memo."
            ),
        },
        peak_table=peak_table,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")

    t_total = time.perf_counter() - t0
    print(f"\nMemo written: {OUTPUT_FILE}")
    print(f"File size:    {OUTPUT_FILE.stat().st_size // 1024} KB")
    print(f"Total time:   {t_total:.1f}s")
    print("\nSummary table:\n")

    display_cols = ["Phase", "2θ (°)", "d-spacing (Å)", "FWHM (°)",
                    "Crystallite Size (nm)", "R²", "Flag"]
    available = [c for c in display_cols if c in summary.columns]
    print(summary[available].to_string(index=False))


if __name__ == "__main__":
    main()
