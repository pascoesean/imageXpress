import numpy as np


class SegmentationModel:
    """
    Abstract base class for 3D segmentation models.
    Subclasses must implement load() and eval()

    Subclasses available:
      - Cellpose2Model ('cellpose2')
      - Cellpose4Model ('cellpose4')
    """

    def __init__(self, anisotropy: float, use_gpu: bool = True):
        self.anisotropy = anisotropy
        self.use_gpu = use_gpu
        self.model = None
        self.load()

    def load(self):
        """Load/initialize the underlying model."""
        raise NotImplementedError

    def eval(self, stack: np.ndarray, diameter_px: float) -> np.ndarray:
        """
        Run inference on a 3D stack (z,y,x).
        Returns an integer label array of the same shape.
        """
        raise NotImplementedError


class Cellpose2Model(SegmentationModel):

    def load(self):
        from cellpose import models
        self.model = models.Cellpose(gpu=self.use_gpu, model_type='nuclei')
        print(f'[Cellpose2Model] loaded (gpu={self.use_gpu})')

    def eval(self, stack: np.ndarray, diameter_px: float) -> np.ndarray:
        masks, _, _, _ = self.model.eval(
            stack,
            do_3D=True,
            anisotropy=self.anisotropy,
            diameter=diameter_px,
            cellprob_threshold=1.0, # tuned
            channels=[0, 0], # gray channel
            z_axis=0,
        )
        return masks


class Cellpose4Model(SegmentationModel):

    def load(self):
        from cellpose import models
        self.model = models.CellposeModel(gpu=self.use_gpu)
        print(f'[Cellpose4Model] loaded (gpu={self.use_gpu})')

    def eval(self, stack: np.ndarray, diameter_px: float) -> np.ndarray:
        masks, _, _, _ = self.model.eval(
            stack,
            do_3D=True,
            anisotropy=self.anisotropy,
            diameter=diameter_px,
            cellprob_threshold=1.0, # tuned
            z_axis=0,
            batch_size=4,
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
