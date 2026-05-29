import tifffile
import glob
import numpy as np
import re
import time
import gc
import torch
from scipy import ndimage
from scipy.ndimage import grey_dilation
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
                      nuclear_channel, diameter, use_gpu, dilation_iterations=5,
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
        channels=[0, 0],  # grayscale
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
    struct = np.ones((1, dilation_iterations * 2 + 1, dilation_iterations * 2 + 1), dtype=bool)
    dilated = grey_dilation(nuclear_masks, footprint=struct)
    cytoplasm_masks = np.where(
        (nuclear_masks == 0) & (dilated != 0),
        dilated,
        0,
    ).astype(nuclear_masks.dtype)
    _elapsed(t, 'cytoplasm dilation')

    # --- Save masks ---
    if save_masks:
        t = time.time()
        tifffile.imwrite(f'{base_path}/{well_id}_nuclear_masks.tif', nuclear_masks.astype(np.uint16))
        tifffile.imwrite(f'{base_path}/{well_id}_cytoplasm_masks.tif', cytoplasm_masks.astype(np.uint16))
        _elapsed(t, 'saving masks')

    _elapsed(t_total, 'TOTAL segment_nuclei_3d')
    return nuclear_masks, cytoplasm_masks


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

    # --- Intensity measurements per channel ---
    t = time.time()
    for channel_idx in range(1, n_channels + 1):
        channel_name = f'channel_{channel_idx}'
        stack = load_channel_stack(base_path, well_id, channel_idx, dtype=np.uint16)
        print(f'  loaded {channel_name} with shape {stack.shape}', flush=True)

        df[f'{channel_name}_nuclear_mean']         = ndimage.mean(stack, labels=nuclear_masks, index=nucleus_ids)
        df[f'{channel_name}_nuclear_max']          = ndimage.maximum(stack, labels=nuclear_masks, index=nucleus_ids)
        df[f'{channel_name}_nuclear_std']          = ndimage.standard_deviation(stack, labels=nuclear_masks, index=nucleus_ids)
        df[f'{channel_name}_nuclear_integrated']   = df[f'{channel_name}_nuclear_mean'] * df['nuclear_volume_voxels']

        df[f'{channel_name}_cytoplasm_mean']       = ndimage.mean(stack, labels=cytoplasm_masks, index=nucleus_ids)
        df[f'{channel_name}_cytoplasm_max']        = ndimage.maximum(stack, labels=cytoplasm_masks, index=nucleus_ids)
        df[f'{channel_name}_cytoplasm_std']        = ndimage.standard_deviation(stack, labels=cytoplasm_masks, index=nucleus_ids)
        df[f'{channel_name}_cytoplasm_integrated'] = df[f'{channel_name}_cytoplasm_mean'] * df['cytoplasm_volume_voxels']

        df[f'{channel_name}_nc_ratio'] = (
            df[f'{channel_name}_nuclear_mean'] / (df[f'{channel_name}_cytoplasm_mean'] + 1e-9)
        )

        del stack
        gc.collect()

    _elapsed(t, 'intensity measurements')

    print(f'MEASUREMENTS FOR {well_id}:', flush=True)
    print(df.head(), flush=True)

    _elapsed(t_total, 'TOTAL measure_intensity')
    return df
