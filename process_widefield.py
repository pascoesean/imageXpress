import sys
import re
from cellpose import models

from functions_widefield import *


# --- Parameters ---
BASE_PATH = Path(sys.argv[1]) # /home/jdweiss1/orcd/scratch/test/15/287-mono-tri-snapshots, etc.
USE_GPU = True


# --- Discover wells ---
wells_found = set()
for f in BASE_PATH.rglob("*.tif"):
    if 'thumb' in f.name.lower():
        continue
    match = re.search(r'_([A-Z]\d{2})(?!\d)', f.name)
    if match:
        wells_found.add(match.group(1))

wells = sorted(wells_found)
print(f"Processing {len(wells)} wells: {wells}")


# load segmentation model
model = models.CellposeModel(gpu=USE_GPU)
print(f'[Cellpose4Model] loaded (gpu={USE_GPU})')


def process_well(well):

    big_img = load_timelapse_stack(well, BASE_PATH)
    num_timepoints = big_img.shape[0]

    mask_stack = []

    for t in range(num_timepoints):
        img = big_img[t,:,:]

        # mask by droplet
        droplet_mask = hough_mask(img)
        img[~droplet_mask] = 0

        # run cellpose
        eeo_masks = run_cellpose(img, model)
        mask_stack.append(eeo_masks)

    # assign track IDs to cells across timepoints
    mask_stack = np.stack(mask_stack, axis=0)
    tracked_stack, merge_df = build_tracked_stack(mask_stack)

    # plot tracked labels overlaid on image for visualization
    save_tracked_masks(big_img, tracked_stack, well, BASE_PATH)

    # extract features from tracked stack
    df = build_measurement_df(mask_stack, tracked_stack)
    #write_df(df, well, BASE_PATH)

    # calculate growth features over time
    growth_df = compute_growth_rates(df, merge_df)
    #write_df(growth_df, well, BASE_PATH, growth=True)

    # label dfs with well_id (useful when concatenating downstream)
    df['well_id'] = well
    growth_df['well_id'] = well

    return df, growth_df



# --- Run sequentially ---
measurements_list = []
failed_wells = []

for i, well in enumerate(wells, start=1):
    try:
        _, measurements = process_well(well)
        measurements_list.append(measurements)
        print(f"[{i}/{len(wells)}] {well} done -> {len(measurements)} EEOs")
    except Exception as e:
        print(f"[{i}/{len(wells)}] {well} FAILED: {e}")
        failed_wells.append(well)

# --- Combine and save ---
all_measurements = pd.concat(measurements_list, ignore_index=True)
all_measurements.to_csv(BASE_PATH / f'all_wells_measurements.csv', index=False)
print(f"\nDone. {len(all_measurements)} total EEOS across {len(wells)} wells.")
print(f"Saved to {BASE_PATH / f'all_wells_measurements.csv'}")
if failed_wells:
    print(f"Failed wells: {failed_wells}")
