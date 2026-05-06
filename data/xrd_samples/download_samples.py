"""
Download 8 XRD patterns from the RRUFF database.

Fetches the bulk XY_RAW.zip archive, extracts files matching target RRUFF IDs,
and saves them to this directory. Run once to populate data/xrd_samples/.

Usage: python data/xrd_samples/download_samples.py
"""
import io
import os
import sys
import zipfile
import requests

BULK_URL = "https://www.rruff.net/zipped_data_files/powder/XY_RAW.zip"

TARGET_IDS = {
    "R040031": "Quartz",
    "R050048": "Calcite",
    "R040096": "Corundum",   # R061220 not present in XY_RAW.zip; R040096 used instead
    "R050115": "Fluorite",
    "R061111": "Magnetite",
    "R050456": "Perovskite", # R060345 not present in XY_RAW.zip; R050456 used instead
    "R040049": "Rutile",
    "R140004": "Kaolinite",  # R061500 not present in XY_RAW.zip; R140004 used instead
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_and_extract() -> None:
    print(f"Downloading {BULK_URL}")
    print("(~88 MB — this may take a moment)")

    resp = requests.get(BULK_URL, timeout=300)
    resp.raise_for_status()

    print(f"Downloaded {len(resp.content) / 1e6:.1f} MB. Extracting...")

    found = {rid: False for rid in TARGET_IDS}

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for entry in zf.namelist():
            for rruff_id, mineral in TARGET_IDS.items():
                if rruff_id in entry and entry.endswith(".txt"):
                    dest = os.path.join(OUTPUT_DIR, f"{mineral}__{rruff_id}.txt")
                    with zf.open(entry) as src:
                        data = src.read()
                    with open(dest, "wb") as dst:
                        dst.write(data)
                    print(f"  Saved: {mineral}__{rruff_id}.txt ({len(data)} bytes)")
                    found[rruff_id] = True
                    break  # take first match per ID

    missing = [f"{TARGET_IDS[rid]} ({rid})" for rid, ok in found.items() if not ok]
    if missing:
        print(f"\nWARNING: Not found in archive: {', '.join(missing)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(TARGET_IDS)} files extracted successfully.")


if __name__ == "__main__":
    download_and_extract()
