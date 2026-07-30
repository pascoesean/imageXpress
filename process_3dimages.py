import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from functions import *
from models import build_model

# --- Parameters ---
BATCH_ID = sys.argv[1]

with open('cellpose_params.json') as f:
    all_params = json.load(f)

if BATCH_ID not in all_params:
    raise KeyError(f"Batch ID '{BATCH_ID}' not found in cellpose_params.json")

p = all_params[BATCH_ID]
BASE_PATH = Path(p['base_path'])
SCALE = p['scale']
XY_PIXEL_UM = p['xy_pixel_um']
DIAMETER_UM = p['diameter_um']
Z_STEP_UM = p['z_step_um']
ANISOTROPY = p['anisotropy']
CELLPROB_THRESHOLD = p['cellprob_threshold']
ACTIN_CHANNEL = p['actin_channel']

N_CHANNELS = 5
NUCLEAR_CHANNEL = 1
USE_GPU = True
MAX_WORKERS = 1

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


# load segmentation model
model = build_model(
    model_type='cellpose2',
    diameter=(DIAMETER_UM / (XY_PIXEL_UM * SCALE)),
    anisotropy=ANISOTROPY,
    cellprob_threshold=CELLPROB_THRESHOLD,
    scale=SCALE,
    use_gpu=USE_GPU
)

# --- Per-well function ---
def process_well(well):
    nuclear_masks, cytoplasm_masks = segment_nuclei_3d(
        well_id=well,
        base_path=str(BASE_PATH),
        nuclear_channel=NUCLEAR_CHANNEL,
        model=model,
        xy_pixel_um=XY_PIXEL_UM,
        z_step_um=Z_STEP_UM,
        actin_channel=ACTIN_CHANNEL
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
all_measurements.to_csv(BASE_PATH / f'all_wells_measurements.csv', index=False)
print(f"\nDone. {len(all_measurements)} total nuclei across {len(wells_to_process)} wells.")
print(f"Saved to {BASE_PATH / f'all_wells_measurements.csv'}")
if failed_wells:
    print(f"Failed wells: {failed_wells}")
