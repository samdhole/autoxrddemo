# XRD Batch Analysis Pipeline

## Goal
Portfolio piece for samdhole.github.io — "Scientific Automation" service lane.
Demonstrates: "A skilled researcher does this in 4 hours. The pipeline runs in 3 minutes."

## Stack
Python | lmfit | scipy | pandas | matplotlib | jinja2 | jupyter

## Mineral Selection (8 RRUFF samples)
| Mineral    | Formula            | Crystal System      | RRUFF ID |
|------------|-------------------|---------------------|----------|
| Quartz     | SiO₂              | Hexagonal           | R040031  |
| Calcite    | CaCO₃             | Trigonal            | R050048  |
| Corundum   | Al₂O₃             | Rhombohedral        | R040096  |
| Fluorite   | CaF₂              | Cubic               | R050115  |
| Magnetite  | Fe₃O₄             | Cubic (spinel)      | R061111  |
| Perovskite | CaTiO₃            | Orthorhombic/Tet    | R050456  |
| Rutile     | TiO₂              | Tetragonal          | R040049  |
| Kaolinite  | Al₂Si₂O₅(OH)₄   | Triclinic           | R140004  |

Note: BaTiO₃ not in RRUFF → use Perovskite (CaTiO₃) R050456
Note: SrTiO₃/tausonite minimal coverage → use Rutile R040049
Note: R061220/R060345/R061500 not in XY_RAW.zip → substituted R040096/R050456/R140004 (same minerals)

## RRUFF File Format
- Headers: lines beginning with `##` (variable count, NOT fixed 11 lines)
- Pattern: `##FIELDNAME=VALUE`
- Key fields: NAMES, RRUFFID, X-RAY WAVELENGTH (may be absent → default 1.54056 Å)
- Data: whitespace or comma-separated `2θ  intensity` pairs
- Bulk download: https://www.rruff.net/zipped_data_files/powder/XY_RAW.zip

## Key Implementation Rules
1. Loader skips all `#`-prefixed lines; never hardcodes line count
2. Wavelength read from header; default 1.54056 Å if absent
3. Scherrer: D = (0.9 × λ) / (β_rad × cos(θ_rad)), θ = peak_center_2θ/2
4. β must be in radians (np.deg2rad(fwhm_deg))
5. Results are upper-bound estimates (instrument broadening not subtracted)
6. lmfit: PseudoVoigtModel, prefix=f'p{i}_', fraction param = η
7. Outlier: Z-score with IQR supplement; both shown in notebook
8. All figures base64-embedded in single HTML file (no external deps)

## Run
```bash
# Download RRUFF data (one-time)
python data/xrd_samples/download_samples.py

# Run tests
pytest tests/

# Execute notebook
jupyter nbconvert --to notebook --execute notebooks/xrd_batch_analysis.ipynb

# Output memo at: output/reports/xrd_memo.html
```

## Known Issues
- Kaolinite R² < 0.95 expected (turbostratic disorder) — not a bug
- With n=8, Z-score > 2.5 rarely triggers → IQR supplement is the useful flag
- lmfit `fraction` param = η (Lorentzian mixing fraction) in PseudoVoigtModel
- RRUFF header line count varies (8–18), never fixed at 11
