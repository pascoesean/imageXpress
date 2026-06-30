import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from pathlib import Path
from skimage.filters import threshold_multiotsu, gaussian
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny, peak_local_max
from skimage.draw import disk
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed



def load_timelapse_stack(donor_id, well_id, exp):
    """
    Load a timelapse stack for a given donor and well.
    """
    base_dir = Path('/home/jdweiss1/orcd/scratch/15/')
    donor_id = str(donor_id)

    # Find snapshot dirs matching this donor
    snapshot_dirs = [
        d for d in base_dir.iterdir()
        if d.is_dir() and 'snapshots' in d.name and donor_id in d.name and exp in d.name
    ]
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directory found for donor {donor_id}")

    frames = []

    for snap_dir in snapshot_dirs:
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
            f"No valid TIF files found for donor={donor_id}, well={well_id}"
        )

    stack = np.stack(frames, axis=0)  # (T, Y, X)
    print(f"Loaded stack: {stack.shape} | dtype: {stack.dtype} | "
          f"donor={donor_id}, well={well_id}")
    return stack


def multiotsu(img):

    thresholds = threshold_multiotsu(img)

    # Using the threshold values, we generate the three regions.
    segmented = np.digitize(img, bins=thresholds)

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    # Plotting the original image.
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(img.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(segmented, cmap='jet')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.savefig('figs_widefield/multiotsu.png')

    return segmented



def hough_mask(img):
    edges = canny(img, sigma=10, low_threshold=10, high_threshold=50)

    plt.imshow(edges)
    plt.savefig('figs_widefield/canny.png')

    hough_radii = np.arange(400, 500, 50)
    hough_res = hough_circle(edges, hough_radii)
    _, cx, cy, radius = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    _, ax = plt.subplots()
    ax.imshow(img)
    c = plt.Circle((cx[0], cy[0]), radius[0], linewidth=2, color='r', fill=False)
    ax.add_patch(c)
    ax.set_axis_off()
    plt.savefig('figs_widefield/hough.png')

    mask = np.zeros(img.shape, dtype=bool)
    rr, cc = disk((cy[0], cx[0]), radius[0], shape=img.shape)
    mask[rr, cc] = True

    return mask


def subtract_bkgrd(img, radius=30):
    _, axs = plt.subplots(1, 3)

    bkgrd = gaussian(img.astype(float), sigma=radius)
    img_corrected = img.astype(float) - bkgrd

    axs[0].imshow(img)
    axs[0].set_axis_off()
    axs[0].set_title('original image')

    axs[1].imshow(bkgrd)
    axs[1].set_axis_off()
    axs[1].set_title('background')

    axs[2].imshow(img_corrected)
    axs[2].set_axis_off()
    axs[2].set_title('corrected image')

    plt.savefig('figs_widefield/blur.png')

    return img_corrected


def measure_eeos(img, mask):
    # EEOs are in class 0 (blue/dark) or class 2 (red) depending on your colormap
    # check which class value corresponds to EEOs
    img[~mask] = 0
    eeo_mask = (img == 0)  # adjust class index

    # remove speckle
    eeo_mask = remove_small_objects(eeo_mask, min_size=100)

    _, ax = plt.subplots()
    ax.imshow(eeo_mask)
    plt.savefig('figs_widefield/eeo_mask.png')

    # label connected components
    labeled = label(eeo_mask)
    props = regionprops_table(labeled, properties=[
        'label',
        'centroid',
        'area',
        'axis_major_length',
        'axis_minor_length',
        'eccentricity',
        'solidity',
        'perimeter',
    ])
    df = pd.DataFrame(props)

    # filter out huge objects (bkgrd)
    df = df[df['area'] <= 10000].reset_index()

    return df


if __name__ == '__main__':
    img = load_timelapse_stack(287, 'B08', 'mono')[0,:,:]
    img = subtract_bkgrd(img)
    mask = hough_mask(img)
    img = multiotsu(img)
    df = measure_eeos(img, mask)
    print(df)
