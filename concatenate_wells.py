#!/usr/bin/env python3
"""
Concatenate timelapse TIFF files by well, sorted by acquisition date.

Usage:
    python concatenate_wells.py --root /Volumes/homes/.../LabUsers/Lauren \
                                --well E09 \
                                --output /path/to/output

File structure expected:
    <root>/<experiment>/<experiment>/<date>/<time>/TimePoint_N/<filename>.tif

Filename pattern:
    <experiment>_<well>_<channel><uuid>.tif

Filters:
    - Keeps only files with "w1" in the filename
    - Skips files containing "thumb" or "w2" (or any other w-channel you specify)
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import tifffile
    import numpy as np
    HAS_TIFF = True
except ImportError:
    HAS_TIFF = False


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate well TIFF files sorted by date.")
    parser.add_argument("--root", required=True,
                        help="Root directory to search (e.g. .../LabUsers/Lauren)")
    parser.add_argument("--well", required=True,
                        help="Well ID to process, e.g. E09. Use 'ALL' to process every well found.")
    parser.add_argument("--output", required=True,
                        help="Output directory for concatenated TIFFs")
    parser.add_argument("--channel", default="w1",
                        help="Channel string to keep (default: w1). Files must contain this string.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing any files")
    return parser.parse_args()


# ── filename / path helpers ───────────────────────────────────────────────────

SKIP_PATTERNS = re.compile(r'thumb|w2|w3|w4|w5', re.IGNORECASE)

def should_skip(filename: str, channel: str) -> bool:
    """Return True if this file should be excluded."""
    if SKIP_PATTERNS.search(filename):
        return True
    if channel.lower() not in filename.lower():
        return True
    return False


def extract_well(filename: str) -> str | None:
    """
    Extract well ID from a filename like:
        022526-ESCMACEEO-LPSP_E09_w1<uuid>.tif
    Returns the well string (e.g. 'E09') or None if not found.
    """
    m = re.search(r'_([A-Z]\d{2})_', filename)
    return m.group(1) if m else None


def extract_date_from_path(path: Path) -> str:
    """
    Walk up the path looking for a folder that matches YYYY-MM-DD.
    Returns the date string, or '' if not found (sorts last).
    """
    for part in path.parts:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', part):
            return part
    return ''


def extract_timepoint(path: Path) -> int:
    """
    Extract TimePoint_N from folder name.  Returns N (int), or 0 if absent.
    """
    for part in path.parts:
        m = re.fullmatch(r'TimePoint_(\d+)', part)
        if m:
            return int(m.group(1))
    return 0


def extract_time_folder(path: Path) -> str:
    """
    The time folder is a 4-digit string like '1739' that sits just below the date folder.
    Returns it for use as a secondary sort key.
    """
    parts = path.parts
    for i, part in enumerate(parts):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', part) and i + 1 < len(parts):
            return parts[i + 1]
    return ''


# ── file discovery ────────────────────────────────────────────────────────────

def find_tif_files(root: str, target_wells: set[str], channel: str) -> dict[str, list[Path]]:
    """
    Recursively search root for .tif files.
    Returns a dict: well_id -> sorted list of Paths.
    """
    well_files: dict[str, list[tuple]] = defaultdict(list)

    for dirpath, _dirs, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith('.tif'):
                continue
            if should_skip(fname, channel):
                continue

            well = extract_well(fname)
            if well is None:
                continue
            if target_wells != {'ALL'} and well not in target_wells:
                continue

            full_path = Path(dirpath) / fname
            date_str  = extract_date_from_path(full_path)
            time_str  = extract_time_folder(full_path)
            tp        = extract_timepoint(full_path)

            well_files[well].append((date_str, time_str, tp, full_path))

    # Sort by (date, time, timepoint) and return just the paths
    sorted_files: dict[str, list[Path]] = {}
    for well, entries in well_files.items():
        entries.sort(key=lambda x: (x[0], x[1], x[2]))
        sorted_files[well] = [e[3] for e in entries]

    return sorted_files


# ── concatenation ─────────────────────────────────────────────────────────────

def concatenate_well(well: str, paths: list[Path], output_dir: Path, dry_run: bool):
    out_file = output_dir / f"{well}_concatenated.tif"

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Well {well}: {len(paths)} frames → {out_file}")
    for i, p in enumerate(paths):
        date = extract_date_from_path(p)
        tp   = extract_timepoint(p)
        print(f"  [{i+1:>3}] {date}  TP{tp:>4}  {p.name}")

    if dry_run:
        return

    if not HAS_TIFF:
        print("ERROR: tifffile is not installed. Run:  pip install tifffile numpy")
        return

    frames = []
    for p in paths:
        img = tifffile.imread(str(p))
        frames.append(img)

    stack = np.stack(frames, axis=0)   # shape: (T, H, W) or (T, H, W, C)
    output_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_file), stack, imagej=True)
    print(f"  Saved {stack.shape} stack to {out_file}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    target_wells = {'ALL'} if args.well.upper() == 'ALL' else {args.well.upper()}
    output_dir   = Path(args.output)

    print(f"Searching: {args.root}")
    print(f"Well(s):   {args.well}")
    print(f"Channel:   {args.channel}")
    print(f"Output:    {output_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be written.\n")

    well_files = find_tif_files(args.root, target_wells, args.channel)

    if not well_files:
        print("No matching files found. Check --root, --well, and --channel.")
        return

    for well, paths in sorted(well_files.items()):
        concatenate_well(well, paths, output_dir, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
