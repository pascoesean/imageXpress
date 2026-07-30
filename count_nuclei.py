import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

from models import build_model
from functions import *

BASE_PATH = Path('/home/jdweiss1/orcd/scratch/12/full-droplet-co1/')
N_CHANNELS = 2
NUCLEAR_CHANNEL = 1
XY_PIXEL_UM = 2.0379
Z_STEP_UM = 5.0
DIAMETER_PX = 10


# --- Discover wells ---
wells_found = defaultdict(set)

for f in BASE_PATH.rglob("*.tif"):
    match = re.search(r'_([A-Z]\d{2})_s(\d+)_w\d', f.name)
    if match:
        well, site = match.group(1), match.group(2)
        wells_found[well].add(f'{well}_s{site}')
    else:
        match = re.search(r'_([A-Z]\d{2})_w\d', f.name)
        if match:
            well = match.group(1)
            wells_found[well]  # ensure key exists even with no site info

# convert to {well: [sorted site strings]}
wells_to_process = {well: sorted(imgs) for well, imgs in wells_found.items()}
print(f"Processing {len(wells_to_process)} wells: {wells_to_process}")


model = build_model(
    model_type='cellpose2',
    diameter=DIAMETER_PX,
    anisotropy=(Z_STEP_UM / XY_PIXEL_UM),
    cellprob_threshold=0.0,
    scale=1,
    use_gpu=True
)


# --- Per-image function ---
def process_image(img):

    print(f'IMG: {img}')

    nuclear_masks, _ = segment_nuclei_3d(
        well_id=img,
        base_path=str(BASE_PATH),
        nuclear_channel=NUCLEAR_CHANNEL,
        model=model,
        xy_pixel_um=XY_PIXEL_UM,
        z_step_um=Z_STEP_UM
    )

    nucleus_ids = np.unique(nuclear_masks)
    nucleus_ids = nucleus_ids[nucleus_ids > 0]
    df = pd.DataFrame({'nucleus_id': nucleus_ids})

    for channel in (1, 2):
        channel_name = f'channel_{channel}'
        stack = load_channel_stack(BASE_PATH, img, channel)
        print(f'  loaded {channel_name} with shape {stack.shape}', flush=True)

        nuc_mean, nuc_max, _ = measure_channel(stack, nuclear_masks, nucleus_ids)
        df[f'{channel_name}_mean'] = nuc_mean
        #df[f'{channel_name}_max'] = nuc_max

    """
    from matplotlib import pyplot as plt
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(df['channel_2_mean'], bins=50)
    ax1.axvline(x=10000, color='red', linestyle='--', label='ESC/Mac threshold')
    ax1.set_xlabel('Intensity')
    ax1.legend()

    ax2.hist(df['channel_2_max'], bins=50)
    ax2.axvline(x=10000, color='red', linestyle='--', label='ESC/Mac threshold')
    ax2.set_xlabel('Intensity')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(BASE_PATH / f'{img}_channel_2_hist.png', dpi=300)
    """

    n_nuclei = nuclear_masks.max()
    n_macs = (df['channel_2_mean'] > 10000).sum()
    n_escs = (df['channel_2_mean'] <= 10000).sum()

    columns = ['img_id', 'n_nuclei', 'n_escs', 'n_macs']
    values = [img, n_nuclei, n_escs, n_macs]
    measurements = pd.DataFrame([values], columns=columns)

    return measurements



# --- Run sequentially ---
measurements_list = []
failed_wells = []

n_wells = len(wells_to_process)

for i, (well, images) in enumerate(wells_to_process.items(), start=1):
    print(f'WELL: {well}')
    #try:
    # build a df from all images/sections in the well
    well_measurements = pd.concat(
        [process_image(img) for img in images],
        ignore_index=True,
    )
    well_measurements['well_id'] = well
    measurements_list.append(well_measurements)
    print(f"[{i}/{n_wells}] {well} done -> {len(well_measurements)} nuclei")
    #except Exception as e:
    #    print(f"[{i}/{n_wells}] {well} FAILED: {e}")
    #    failed_wells.append(well)

# --- Combine and save ---
all_measurements = pd.concat(measurements_list, ignore_index=True)
all_measurements.to_csv(BASE_PATH / f'all_wells_measurements.csv', index=False)
print(f"\nDone. {len(all_measurements)} total nuclei across {len(wells_to_process)} wells.")
print(f"Saved to {BASE_PATH / f'all_wells_measurements.csv'}")
if failed_wells:
    print(f"Failed wells: {failed_wells}")