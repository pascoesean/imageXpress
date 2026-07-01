import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from pathlib import Path
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import disk
from skimage.measure import regionprops_table



def load_timelapse_stack(well_id, snap_dir):
    """
    Load a timelapse stack for a given donor and well.
    """
    frames = []

    # Each subfolder is a date-named acquisition folder
    for acq_dir in sorted(snap_dir.iterdir()):
        if not acq_dir.is_dir():
            continue

        # Walk all TimePoint_N folders underneath
        for tp_dir in sorted(acq_dir.rglob('TimePoint_*')):
            if not tp_dir.is_dir():
                continue

            # Find tifs for this well, excluding thumbs
            tifs = sorted([
                f for f in tp_dir.glob('*.tif')
                if well_id in f.name and 'thumb' not in f.name.lower()
            ])

            if not tifs:
                continue

            # If multiple tifs per timepoint (channels/z), stack and max-project or pick first
            imgs = [tifffile.imread(str(f)) for f in tifs]
            frame = imgs[0]
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No valid TIF files found for well={well_id}"
        )

    stack = np.stack(frames, axis=0)  # (T, Y, X)
    print(f"Loaded stack: {stack.shape} | dtype: {stack.dtype} | "
          f"well={well_id}")
    return stack


def hough_mask(img):
    """
    Uses a circle Hough transform to identify the perimeter of the droplet.
    We can apply the droplet mask to filter out background junk!
    """
    # canny transform -> edges
    edges = canny(img, sigma=10, low_threshold=10, high_threshold=50)

    # hough transform -> droplet perimeter
    hough_radii = np.arange(400, 500, 50)
    hough_res = hough_circle(edges, hough_radii)
    _, cx, cy, radius = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    # create mask for interior of droplet
    mask = np.zeros(img.shape, dtype=bool)
    rr, cc = disk((cy[0], cx[0]), radius[0], shape=img.shape)
    mask[rr, cc] = True

    return mask


def run_cellpose(img, model):
    """
    Run inference on Cellpose4 model.
    """
    eeo_masks, _, _ = model.eval(img)
    return eeo_masks



def link_and_relabel(mask_prev, mask_curr, next_track_id, overlap_thresh=0.1):
    """
    Forward-propagates labels from mask_prev to mask_curr, such that IDs
    remain consistent for objects through all five timepoints. Assigns
    curr_obj the ID of the prev_obj with greatest IOU (ignores IOU < overlap_thresh).

    Returns: relabeled_curr, merge_events (list of dicts), next_track_id
    """
    relabeled = np.zeros_like(mask_curr)
    merge_events = []

    curr_labels = np.unique(mask_curr)
    curr_labels = curr_labels[curr_labels != 0]

    for lbl in curr_labels:
        curr_obj = mask_curr == lbl
        curr_area = curr_obj.sum()

        # get prev_labels that overlap with curr_obj
        overlapping_labels, counts = np.unique(mask_prev[curr_obj], return_counts=True)
        keep = overlapping_labels != 0
        overlapping_labels, counts = overlapping_labels[keep], counts[keep]

        if len(overlapping_labels) == 0:
            # no parent -> new track (entered frame, e.g. drifted into droplet)
            relabeled[curr_obj] = next_track_id
            next_track_id += 1
            continue

        # iou = (curr_area & prev_area) / (curr_area | prev_area)
        ious = np.array([
            count / (curr_area + (mask_prev == prev_label).sum() - count)
            for prev_label, count in zip(overlapping_labels, counts)
        ])

        # ignore objects with ious < overlap_thresh
        valid = ious >= overlap_thresh
        if not valid.any():
            relabeled[curr_obj] = next_track_id
            next_track_id += 1
            continue

        parents = overlapping_labels[valid]
        parent_ious = ious[valid]

        # choose parent with highest iou
        winner = parents[np.argmax(parent_ious)]
        relabeled[curr_obj] = winner

        if len(parents) > 1:
            merge_events.append({
                'surviving_track': int(winner),
                'merged_tracks': [int(p) for p in parents if p != winner],
            })

    return relabeled, merge_events, next_track_id


def build_tracked_stack(mask_stack, overlap_thresh=0.1):
    """
    mask_stack: (T, Y, X) array of per-frame instance label masks
    (e.g. from your run_cellpose loop, stacked along axis 0 — relabel
    each frame with skimage.measure.label first if cellpose labels
    aren't already unique/connected per frame).

    Returns: tracked_stack (T, Y, X) with consistent track IDs across time,
             merge_log: DataFrame with columns [t, surviving_track, merged_tracks]
    """
    T = mask_stack.shape[0]
    tracked = np.zeros_like(mask_stack)
    tracked[0] = mask_stack[0]

    next_track_id = int(mask_stack[0].max()) + 1
    merge_log = []

    for t in range(1, T):
        # run relabeling on each (x, y) image
        relabeled, merges, next_track_id = link_and_relabel(
            tracked[t - 1], mask_stack[t], next_track_id, overlap_thresh
        )
        tracked[t] = relabeled
        for m in merges:
            m['t'] = t
            merge_log.append(m)

    merge_df = pd.DataFrame(merge_log, columns=['t', 'surviving_track', 'merged_tracks'])
    return tracked, merge_df



def measure_tracked_frame(img, tracked_mask, t):
    props = regionprops_table(
        tracked_mask,
        intensity_image=img,
        properties=('label', 'centroid', 'area', 'eccentricity',
                    'solidity', 'perimeter', 'image_intensity'),
    )
    df = pd.DataFrame(props)
    df = df.rename(columns={'centroid-0': 'centroid_y', 'centroid-1': 'centroid_x',
                             'label': 'track_id'})
    df['circularity'] = 4 * np.pi * df['area'] / (df['perimeter'] ** 2 + 1e-8)
    df['intensity_std'] = df['image_intensity'].apply(lambda im: im[im > 0].std())
    df['equiv_diameter'] = 2 * np.sqrt(df['area'] / np.pi)
    df['t'] = t
    df = df.drop(columns='image_intensity')
    return df



def build_measurement_df(img_stack, tracked_stack):
    """
    img_stack, tracked_stack: (T, Y, X)
    Returns long-format df: one row per (track_id, t).
    """
    frames = [
        measure_tracked_frame(img_stack[t], tracked_stack[t], t)
        for t in range(tracked_stack.shape[0])
    ]
    return pd.concat(frames, ignore_index=True)



def compute_growth_rates(growth_df, merge_df, min_threshold=3):
    """
    Fits a line to equiv_diameter vs. t per track. Flags tracks that
    experienced a merge.

    One row per track_id.
    """
    merged_tracks = set(merge_df['surviving_track']) if len(merge_df) else set()

    rows = []
    for track_id, grp in growth_df.groupby('track_id'):
        grp = grp.sort_values('t')

        # throw out single-timepoint objects (bubbles, debris, etc.)
        if len(grp) < min_threshold:
            continue

        slope_diam, _ = np.polyfit(grp['t'], grp['equiv_diameter'], 1)
        slope_area, _ = np.polyfit(grp['t'], grp['area'], 1)

        rows.append({
            'track_id': track_id,
            'n_timepoints': len(grp),
            't_start': grp['t'].min(),
            't_end': grp['t'].max(),
            'diameter_start': grp['equiv_diameter'].iloc[0],
            'diameter_end': grp['equiv_diameter'].iloc[-1],
            'diameter_growth_rate': slope_diam,      # px (or um, if you scale t/area first) per timepoint
            'area_start': grp['area'].iloc[0],
            'area_end': grp['area'].iloc[1],
            'area_growth_rate': slope_area,
            'had_merge': track_id in merged_tracks,
        })

    return pd.DataFrame(rows)



def save_tracked_masks(img_stack, tracked_stack, well, base_path):
    """
    Plots the tracked labels overlaid on the original image.
    """
    num_timepoints = img_stack.shape[0]
    fig, axs = plt.subplots(1, num_timepoints)

    # build a consistent colormap keyed to track_id so the same EEO is the
    # same color in every panel
    max_track_id = int(tracked_stack.max())
    rng = np.random.default_rng(0)  # fixed seed -> reproducible colors across runs
    colors = rng.random((max_track_id + 1, 3))
    colors[0] = 0  # background stays black/transparent

    for t, ax in enumerate(axs):
        img = img_stack[t, :, :]
        mask = tracked_stack[t, :, :]

        ax.imshow(img, cmap='gray')

        # mask out background so it doesn't tint the whole image
        overlay = np.ma.masked_where(mask == 0, mask)
        ax.imshow(colors[overlay], alpha=0.45)

        ax.set_title(f't={t}')
        ax.set_axis_off()

    fig.tight_layout()
    plt.savefig(f'{base_path}/masks/{well}.png', dpi=150)



def plot_track(df, track_id=None):
    """
    Plots the position & growth of one track over 5 days of measurement.
    """
    if track_id is None:
        filter = df['n_timepoints'] == 5
        track_id = df['track_id'][filter].sample(n=1)

    filter = df['track_id'] == track_id
    x = df['centroid_x'][filter]
    y = df['centroid_y'][filter]
    area = df['area'][filter]
    t = list(range(1, 6))

    fig, axs = plt.subplots(1, 2)

    axs[0].plot(x, y)
    axs[0].xlabel('x')
    axs[0].ylabel('y')

    axs[1].plot(t, area)
    axs[1].xlabel('t')
    axs[1].ylabel('area')

    fig.tight_layout()
    plt.savefig(f'figs/track_{track_id}.png')



def write_df(df, well, base_path, growth=False):
    if growth:
        file = f'{base_path}/{well}_growth_df.csv'
    else:
        file = f'{base_path}/{well}_df.csv'
    df.to_csv(file)
