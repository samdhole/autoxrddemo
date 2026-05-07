from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class XRDData:
    name: str
    path: Path
    metadata: dict
    df: pd.DataFrame  # columns: two_theta, intensity (normalized 0-100)
    wavelength: float  # Å


class XRDLoader:
    DEFAULT_WAVELENGTH = 1.54056  # Cu Kα
    MAX_POINTS = 4000  # 0.02 deg/pt typical lab step; ROI-fit handles density fine

    @staticmethod
    def load(path: str | Path, name: str | None = None) -> XRDData:
        path = Path(path)
        name = name or path.stem
        metadata: dict = {}
        data_lines: list[tuple[float, float]] = []

        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("##"):
                    key, _, val = line[2:].partition("=")
                    metadata[key.strip()] = val.strip()
                elif line.startswith("#"):
                    pass
                else:
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 2:
                        try:
                            data_lines.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass

        df = pd.DataFrame(data_lines, columns=["two_theta", "intensity"])
        if df.empty:
            raise ValueError(f"No XRD data rows found in {path}")
        max_i = df["intensity"].max()
        if max_i > 0:
            df["intensity"] = df["intensity"] / max_i * 100.0

        # Resample to max_points via uniform stride; 600 pts is sufficient for
        # PseudoVoigt peak fitting and keeps fit time under ~2s per sample.
        if len(df) > XRDLoader.MAX_POINTS:
            step = len(df) // XRDLoader.MAX_POINTS
            df = df.iloc[::step].reset_index(drop=True)

        # Try multiple likely wavelength field names
        wl_raw = (
            metadata.get("X-RAY WAVELENGTH")
            or metadata.get("WAVELENGTH")
            or metadata.get("X RAY WAVELENGTH")
        )
        try:
            wavelength = float(wl_raw) if wl_raw else XRDLoader.DEFAULT_WAVELENGTH
        except (ValueError, TypeError):
            wavelength = XRDLoader.DEFAULT_WAVELENGTH

        return XRDData(
            name=name,
            path=path,
            metadata=metadata,
            df=df,
            wavelength=wavelength,
        )

    @staticmethod
    def load_directory(dirpath: str | Path) -> dict[str, XRDData]:
        dirpath = Path(dirpath)
        result = {}
        for p in sorted(dirpath.glob("*.txt")):
            if p.name.startswith("download"):
                continue
            xrd = XRDLoader.load(p)
            result[xrd.name] = xrd
        return result
