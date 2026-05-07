"""
Generates 8 synthetic quartz XRD patterns with systematic FWHM broadening.

Purpose: demo dataset for the same-material batch trend analysis section of the
portfolio notebook. Simulates a quartz annealing series where grain refinement
causes peak broadening (FWHM 0.08 -> 0.22 deg across 8 samples).

Peak positions and relative intensities based on ICDD 00-046-1045 (alpha-quartz,
Cu Kalpha1 radiation, lambda=1.54056 A). Gaussian peak profiles, SNR ~100
(noise_std = peak_max / 100).

Run:  python generate_synthetic.py
Output: Quartz_Anneal_01.txt ... Quartz_Anneal_08.txt (RRUFF-format headers)
"""
from pathlib import Path
import numpy as np

QUARTZ_PEAKS = [
    (20.86, 17),
    (26.65, 100),
    (36.54, 8),
    (39.47, 7),
    (40.29, 6),
    (42.45, 8),
    (45.79, 4),
    (50.14, 13),
    (54.87, 4),
    (55.33, 4),
    (59.96, 5),
    (64.02, 3),
    (67.74, 6),
    (68.14, 5),
    (73.47, 4),
]

OUT_DIR = Path(__file__).parent
rng = np.random.default_rng(42)
x = np.linspace(10.0, 80.0, 3500)  # 0.02 deg/pt — typical lab step size
fwhm_values = np.linspace(0.08, 0.22, 8)


def gaussian(x, center, amplitude, fwhm):
    sigma = fwhm / 2.3548
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


for i, fwhm in enumerate(fwhm_values, start=1):
    y = np.ones_like(x) * 2.0
    for center, rel_int in QUARTZ_PEAKS:
        y += gaussian(x, center, rel_int, fwhm)
    noise_std = y.max() / 100.0  # SNR=100 (typical modern lab XRD; lets auto-detection catch ~3% reflections)
    y += rng.normal(0, noise_std, size=len(x))
    y = np.clip(y, 0, None)
    y = y / y.max() * 100.0

    fname = OUT_DIR / f"Quartz_Anneal_{i:02d}.txt"
    lines = [
        f"##NAMES=Quartz Annealed Sample {i:02d}\n",
        f"##RRUFFID=SYN{i:06d}\n",
        f"##SOURCE=Synthetic — systematic FWHM broadening demo\n",
        f"##X-RAY WAVELENGTH=1.54056\n",
        f"##FWHM_NOMINAL={fwhm:.4f}\n",
        f"##END=\n",
    ]
    for xi, yi in zip(x, y):
        lines.append(f"{xi:.4f}, {yi:.4f}\n")
    fname.write_text("".join(lines), encoding="utf-8")

print(f"Generated {len(fwhm_values)} synthetic quartz patterns in {OUT_DIR}")
