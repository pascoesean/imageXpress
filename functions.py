import tifffile
import glob
import numpy as np
import re
import time
import gc
import torch
from scipy import ndimage
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree
from skimage.measure import regionprops, marching_cubes, mesh_surface_area
from skimage.transform import downscale_local_mean, resize
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def _elapsed(start, label):
    print(f'  [{label}] {time.time() - start:.1f}s', flush=True)


# cached Cellpose model for this process
_CELLPOSE_MODEL = None


def _channel_file_paths(base_path, well_id, channel_index):
    z_planes = sorted(
        glob.glob(f'{base_path}/ZStep_*/*_{well_id}_w{channel_index}*.tif'),
        key=lambda f: int(re.search(r'ZStep_(\d+)', f).group(1))
    )
    return [p for p in z_planes if not re.search(r'thumb', p)]


def load_channel_stack(base_path, well_id, channel_index, dtype=None):
    paths = _channel_file_paths(base_path, well_id, channel_index)
    if len(paths) == 0:
        return np.empty((0, 0, 0), dtype=np.float32 if dtype is None else dtype)
    sample = tifffile.imread(paths[0])
    if dtype is None:
        dtype = sample.dtype
    arr = np.empty((len(paths),) + sample.shape, dtype=dtype)
    arr[0] = sample.astype(dtype, copy=False)
    for zi, fpath in enumerate(paths[1:], start=1):
        arr[zi] = tifffile.imread(fpath).astype(dtype, copy=False)
    return arr


def get_cellpose_model(use_gpu):
    """Return a cached Cellpose 2.x model (nuclei) for this process."""
    global _CELLPOSE_MODEL
    if _CELLPOSE_MODEL is None:
        from cellpose import models
        _CELLPOSE_MODEL = models.Cellpose(gpu=use_gpu, model_type='nuclei')
        device = next(_CELLPOSE_MODEL.cp.net.parameters()).device
        print(f'  Cellpose model loaded on {device}', flush=True)
    return _CELLPOSE_MODEL


def segment_nuclei_3d(well_id, base_path, n_channels, z_step_um, xy_pixel_um,
                      nuclear_channel, diameter, use_gpu, dilation_iterations=10,
                      scale=1, save_masks=True, model=None):

    t_total = time.time()

    # --- Load nuclear channel ---
    t = time.time()
    nuclear_stack = load_channel_stack(base_path, well_id, nuclear_channel, dtype=np.float32)
    print(f'  nuclear channel loaded with shape {nuclear_stack.shape}', flush=True)
    _elapsed(t, 'image loading')

    model_to_use = model if model is not None else get_cellpose_model(use_gpu)

    # --- Downsample XY for faster inference ---
    t = time.time()
    print(f'Segmenting nuclei for {well_id} (scale={scale})...', flush=True)
    print(f'  GPU memory before eval: {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    nuclear_stack_ds = downscale_local_mean(nuclear_stack, (1, scale, scale)).astype(np.float32)
    print(f'  downsampled stack shape: {nuclear_stack_ds.shape}', flush=True)

    masks_ds, _, _, diams = model_to_use.eval(
        nuclear_stack_ds,
        do_3D=True,
        anisotropy=z_step_um / (xy_pixel_um * scale),
        diameter=diameter / scale,
        cellprob_threshold=2.0,
        channels=[0,0], # grayscale
        z_axis=0,
    )

    print(f'  estimated diameter by model: {diams:.1f}px', flush=True)
    print(f'  GPU memory after eval: {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    # --- Scale masks back to original XY resolution ---
    nuclear_masks = resize(
        masks_ds.astype(np.float32),
        nuclear_stack.shape,
        order=0,               # nearest-neighbor preserves label IDs
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.uint16)

    print(f'  {nuclear_masks.max()} nuclei detected.', flush=True)
    _elapsed(t, 'cellpose nuclear segmentation')
    del nuclear_stack, nuclear_stack_ds, masks_ds
    gc.collect()

    # --- Build cytoplasm masks by single-pass dilation ---
    print(f'Building cytoplasm masks for {well_id}...', flush=True)
    t = time.time()
    foreground = nuclear_masks > 0
    struct = np.ones((1, dilation_iterations * 2 + 1, dilation_iterations * 2 + 1), dtype=bool)
    dilated_fg = binary_dilation(foreground, structure=struct)

    # propagate nearest nucleus label onto the dilated ring
    nearest_label_idx = ndimage.distance_transform_edt(
        ~foreground, return_distances=False, return_indices=True
    )
    dilated_labels = nuclear_masks[tuple(nearest_label_idx)]

    # generate cytoplasm masks
    cytoplasm_masks = np.where(
        dilated_fg & ~foreground,
        dilated_labels,
        0,
    ).astype(nuclear_masks.dtype)

    _elapsed(t, 'cytoplasm dilation')

    # --- Save masks ---
    if save_masks:
        t = time.time()
        tifffile.imwrite(f'{base_path}/masks/{well_id}_nuclear_masks.tif', nuclear_masks.astype(np.uint16))
        tifffile.imwrite(f'{base_path}/masks/{well_id}_cytoplasm_masks.tif', cytoplasm_masks.astype(np.uint16))
        _elapsed(t, 'saving masks')

    _elapsed(t_total, 'TOTAL segment_nuclei_3d')
    return nuclear_masks, cytoplasm_masks



def calculate_metrics(nuclear_masks, cytoplasm_masks, base_path, n_channels,
                      well_id, z_step_um, xy_pixel_um):

    intensity_df = measure_intensity(nuclear_masks, cytoplasm_masks, base_path, n_channels, well_id, z_step_um, xy_pixel_um)
    morpho_df = measure_morphology(nuclear_masks, well_id, z_step_um, xy_pixel_um)
    df = intensity_df.merge(morpho_df, on='nucleus_id')

    return df



def measure_intensity(nuclear_masks, cytoplasm_masks, base_path, n_channels,
                      well_id, z_step_um, xy_pixel_um):

    t_total = time.time()
    nucleus_ids = np.unique(nuclear_masks)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]
    voxel_volume_um3 = z_step_um * xy_pixel_um * xy_pixel_um

    df = pd.DataFrame({'nucleus_id': nucleus_ids})

    # --- Volumes ---
    t = time.time()
    df['nuclear_volume_voxels'] = ndimage.sum(
        np.ones_like(nuclear_masks), labels=nuclear_masks, index=nucleus_ids
    )
    df['nuclear_volume_um3'] = df['nuclear_volume_voxels'] * voxel_volume_um3
    df['cytoplasm_volume_voxels'] = ndimage.sum(
        np.ones_like(cytoplasm_masks), labels=cytoplasm_masks, index=nucleus_ids
    )
    df['cytoplasm_volume_um3'] = df['cytoplasm_volume_voxels'] * voxel_volume_um3
    _elapsed(t, 'volume measurements')

    # save vals to prevent repeatedly calculating in loop
    nuc_vols = df['nuclear_volume_voxels'].to_numpy()
    cyt_vols = df['cytoplasm_volume_voxels'].to_numpy()

    # --- Intensity measurements per channel ---
    t = time.time()
    for channel_idx in range(1, n_channels + 1):
        channel_name = f'channel_{channel_idx}'
        stack = load_channel_stack(base_path, well_id, channel_idx, dtype=np.uint16)
        print(f'  loaded {channel_name} with shape {stack.shape}', flush=True)

        # call helper function to efficiently get all stats
        nuc_mean, nuc_max, nuc_std = _measure_channel(stack, nuclear_masks, nucleus_ids)
        cyt_mean, cyt_max, cyt_std = _measure_channel(stack, cytoplasm_masks, nucleus_ids)

        df[f'{channel_name}_nuclear_mean']         = nuc_mean
        df[f'{channel_name}_nuclear_max']          = nuc_max
        df[f'{channel_name}_nuclear_std']          = nuc_std
        df[f'{channel_name}_nuclear_integrated']   = nuc_mean * nuc_vols

        df[f'{channel_name}_cytoplasm_mean']       = cyt_mean
        df[f'{channel_name}_cytoplasm_max']        = cyt_max
        df[f'{channel_name}_cytoplasm_std']        = cyt_std
        df[f'{channel_name}_cytoplasm_integrated'] = cyt_mean * cyt_vols

        df[f'{channel_name}_nc_ratio'] = (
            nuc_mean / (cyt_mean + 1e-9)
        )

        del stack
        gc.collect()

    _elapsed(t, 'intensity measurements')

    print(f'MEASUREMENTS FOR {well_id}:', flush=True)
    print(df.head(), flush=True)

    _elapsed(t_total, 'TOTAL measure_intensity')
    return df


def _measure_channel(stack, mask, nucleus_ids):
    """
    Helper function. Calculates intensity metrics in a single pass (groups by label).
    """
    flat_vals = stack.ravel().astype(np.float32)
    flat_labels = mask.ravel()

    # sort by label
    order = np.argsort(flat_labels, kind='stable')
    sorted_labels = flat_labels[order]
    sorted_vals = flat_vals[order]

    # find boundaries btwn labels
    unique_labels, counts = np.unique(sorted_labels, return_counts=True)

    splits = np.cumsum(counts)[:-1]
    groups = np.split(sorted_vals, splits)

    label_to_stats = {}
    for lbl, grp in zip(unique_labels, groups):
        if lbl == 0:
            continue
        mean = grp.mean()
        label_to_stats[lbl] = (mean, grp.max(), grp.std())

    means = np.array([label_to_stats.get(i, (0.0, 0.0, 0.0))[0] for i in nucleus_ids])
    maxs = np.array([label_to_stats.get(i, (0.0, 0.0, 0.0))[1] for i in nucleus_ids])
    stds = np.array([label_to_stats.get(i, (0.0, 0.0, 0.0))[2] for i in nucleus_ids])

    return means, maxs, stds



def measure_morphology(mask, well_id, z_step_um, xy_pixel_um, radius_um=50):
    """
    Helper function. Calculates morphology metrics in a single pass (groups by label).
    """
    nucleus_ids = np.unique(mask)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]
    voxel_volume_um3 = z_step_um * xy_pixel_um * xy_pixel_um

    df = pd.DataFrame({'nucleus_id': nucleus_ids})

    t_total = time.time()

    props = regionprops(mask, spacing=(z_step_um, xy_pixel_um, xy_pixel_um)) # note: 3D mask

    # regionprops returns one object per label, in label order but not guaranteed
    # to match nucleus_ids ordering — so index by label explicitly
    prop_by_label = {p.label: p for p in props}

    centroids = np.array([prop_by_label[i].centroid for i in nucleus_ids])
    df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]] = centroids

    df['axis_major_length'] = np.array([prop_by_label[i].axis_major_length for i in nucleus_ids])
    df['axis_minor_length'] = np.array([prop_by_label[i].axis_minor_length for i in nucleus_ids])

    _elapsed(t_total, 'regionprops measurements')

    t = time.time()
    # use a kd-tree to calculate local density metrics
    kdtree = KDTree(centroids)

    # query tree for info on 5 nearest neighbors
    neighbor_dists, neighbor_idx = kdtree.query(centroids, k=5)

    # note: the 0th item in each list is a self-node --> start indexing at 1
    df['nn_dist'] = neighbor_dists[:, 1]
    df['nn_nucleus_id'] = nucleus_ids[neighbor_idx[:, 1]]
    df['nn5_mean_dist'] = neighbor_dists[:, 1:].mean(axis=1)

    # query tree for # neighbors within radius
    neighbor_idx = kdtree.query_ball_point(centroids, radius_um)
    df[f'num_neighbors_within_{int(radius_um)}_um'] = [len(nb) - 1 for nb in neighbor_idx]

    _elapsed(t, 'local density measurements')

    # run marching cubes algorithm to generate 3D model of each nucleus
    sphericities = []
    t = time.time()

    for nid in nucleus_ids:
        # first, reduce the search space to the nucleus' bounding box (from regionprops)
        p = prop_by_label[nid]
        slices = p.slice

        # create a binary mask for the specific nucleus
        binary_mask = (mask[slices] == nid)
        # pad by 1 pixel so that mesh is closed on ends
        binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)

        try:
            verts, faces, _, _ = marching_cubes(binary_mask, level=0.5, spacing=(z_step_um, xy_pixel_um, xy_pixel_um))
            surface_area = mesh_surface_area(verts, faces)
            # note: num_pixels property is equivalent to num_voxels for 3D images
            volume = p.num_pixels * voxel_volume_um3
            # calculate sphericity (how close is surface area of the 3D object to the surface area of a sphere?)
            sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
        except Exception:
            # marching_cubes can fail for very small/irregular volumes
            sphericity = np.nan

        sphericities.append(sphericity)

    df['sphericity'] = np.array(sphericities)
    _elapsed(t, 'marching_cubes measurements')

    print(f'MEASUREMENTS FOR {well_id}:', flush=True)
    print(df.head(), flush=True)

    _elapsed(t_total, 'TOTAL measure_morphology')

    return df
