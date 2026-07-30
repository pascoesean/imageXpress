import numpy as np
from cellpose import models
from skimage.transform import downscale_local_mean, resize


class SegmentationModel:
    """
    Abstract base class for 3D segmentation models.
    Subclasses must implement load() and eval()

    Subclasses available:
      - Cellpose2Model ('cellpose2')
      - Cellpose4Model ('cellpose4')
    """

    def __init__(self, diameter: float, anisotropy: float, cellprob_threshold: float, scale: float, use_gpu: bool = True):
        self.diameter = diameter
        self.anisotropy = anisotropy
        self.cellprob_threshold = cellprob_threshold
        self.scale = scale
        self.use_gpu = use_gpu
        self.model = None
        self.load()

    def load(self):
        """Load/initialize the underlying model."""
        raise NotImplementedError

    def eval(self, stack: np.ndarray) -> np.ndarray:
        """
        Run inference on a 3D stack (z,y,x).
        Returns an integer label array of the same shape.
        """
        raise NotImplementedError


class Cellpose2Model(SegmentationModel):

    def load(self):
        self.model = models.Cellpose(gpu=self.use_gpu, model_type='nuclei')
        print(f'[Cellpose2Model] loaded (gpu={self.use_gpu})')

    def eval(self, stack: np.ndarray) -> np.ndarray:
        """
        Wrapper for CellposeModel.eval
        """
        orig_shape = stack.shape

        if self.scale != 1:
            # downsample xy for faster inference
            stack = downscale_local_mean(stack, (1, self.scale, self.scale)).astype(np.float32)
            print(f'  downsampled stack shape: {stack.shape}', flush=True)

        masks, _, _, _ = self.model.eval(
            stack,
            do_3D=True,
            diameter=self.diameter,
            anisotropy=self.anisotropy,
            cellprob_threshold=self.cellprob_threshold,
            channels=[0, 0], # gray channel
            z_axis=0,
        )

        # restore original (z, y, x) dims in one nearest-neighbor resize
        masks = resize(
            masks.astype(np.float32),
            orig_shape,
            order=0, # maintain nearest labels
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.uint16)

        del stack

        return masks


class Cellpose4Model(SegmentationModel):

    def load(self):
        self.model = models.CellposeModel(gpu=self.use_gpu)
        print(f'[Cellpose4Model] loaded (gpu={self.use_gpu})')

    def eval(self, stack: np.ndarray) -> np.ndarray:
        masks, _, _ = self.model.eval(
            stack,
            do_3D=False,
            stitch_threshold=0.25,
            diameter=self.diameter,
            anisotropy=self.anisotropy,
            cellprob_threshold=self.cellprob_threshold,
            z_axis=0,
        )
        return masks


_REGISTRY = {
    'cellpose2': Cellpose2Model,
    'cellpose4': Cellpose4Model,
}

def build_model(model_type: str, **kwargs) -> SegmentationModel:
    """
    Returns an initialized model ready for inference.
    """
    key = model_type.lower()
    if key not in _REGISTRY:
        raise ValueError(f'Unknown model type: {model_type}. Choose from {list(_REGISTRY)}')
    return _REGISTRY[key](**kwargs)
