import re
import sys
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path

from functions import segment_nuclei_3d, calculate_metrics, get_cellpose_model

# --- Parameters ---
BASE_PATH = Path(sys.argv[1])
SCALE = 2
N_CHANNELS = 5
Z_STEP_UM = 5.0
XY_PIXEL_UM = 0.6793
NUCLEAR_CHANNEL = 1
DIAMETER = 20
USE_GPU = True
MAX_WORKERS = 1  # TODO: try processing wells in parallel


# --- Discover wells ---
wells_found = set()
for f in BASE_PATH.rglob("*.tif"):
    match = re.search(r'_([A-Z]\d{2})_w\d', f.name)
    if match:
        wells_found.add(match.group(1))
    elif re.search(r'_([A-Z]\d{2}_s\d{1})_w\d', f.name):
        match = re.search(r'_([A-Z]\d{2}_s\d{1})_w\d', f.name)
        wells_found.add(match.group(1))

wells_to_process = sorted(wells_found)
print(f"Processing {len(wells_to_process)} wells: {wells_to_process}")


model = get_cellpose_model(USE_GPU)

# --- Per-well function ---
def process_well(well):
    nuclear_masks, cytoplasm_masks = segment_nuclei_3d(
        well_id=well,
        base_path=str(BASE_PATH),
        n_channels=N_CHANNELS,
        z_step_um=Z_STEP_UM,
        xy_pixel_um=XY_PIXEL_UM,
        scale=SCALE,  # downscale for faster processing and lower GPU memory usage
        nuclear_channel=NUCLEAR_CHANNEL,
        diameter=DIAMETER,
        use_gpu=USE_GPU,
        model=model
    )

    measurements = calculate_metrics(
        nuclear_masks=nuclear_masks,
        cytoplasm_masks=cytoplasm_masks,
        base_path=str(BASE_PATH),
        n_channels=N_CHANNELS,
        well_id=well,
        z_step_um=Z_STEP_UM,
        xy_pixel_um=XY_PIXEL_UM
    )

    measurements['well_id'] = well
    return well, measurements


# --- Run sequentially ---
measurements_list = []
failed_wells = []

for i, well in enumerate(wells_to_process, start=1):
    try:
        _, measurements = process_well(well)
        measurements_list.append(measurements)
        print(f"[{i}/{len(wells_to_process)}] {well} done -> {len(measurements)} nuclei")
    except Exception as e:
        print(f"[{i}/{len(wells_to_process)}] {well} FAILED: {e}")
        failed_wells.append(well)

# --- Combine and save ---
all_measurements = pd.concat(measurements_list, ignore_index=True)
all_measurements.to_csv(BASE_PATH / 'all_wells_measurements.csv', index=False)
print(f"\nDone. {len(all_measurements)} total nuclei across {len(wells_to_process)} wells.")
print(f"Saved to {BASE_PATH / 'all_wells_measurements.csv'}")
if failed_wells:
    print(f"Failed wells: {failed_wells}")
