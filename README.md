# Endometrial Co-Culture IF Toolbox

A pipeline for segmenting, measuring, and classifying cells in 3D immunofluorescence images of endometrial co-culture and tri-culture systems — built for studying macrophage–stromal–epithelial interactions in the context of endometriosis.

## Features

- **3D nuclear & cytoplasm segmentation** of dense organoid structures via Cellpose
- **Random forest cell-type classification** across mono-, co-, and tri-culture conditions
- Slurm-based batch processing
- Per-well, per-donor morphology, intensity, and density feature extraction

## Experiments

| ID | Conditions | Donors |
|----|------------|--------|
| **12** | ESC (mono), EEO+Mac (co), Mac (mono) | 1 EEO/ESC donor |
| **14** | EEO (mono), EEO+ESC (co), EEO+ESC+macrophage (tri) | 1 EEO/ESC donor |
| **15** | EEO (mono), EEO+ESC (co), EEO+ESC+macrophage (tri) | 3 EEO/ESC donors |

## Pipeline Structure

### Segmentation

| File | Purpose |
|------|---------|
| `cellpose_gpu.sh` | Slurm script — passes parameters from `cellpose_params.json` to `process_3d_images.py` |
| `cellpose_params.json` | Processing parameters for nuclear and cytoplasm segmentation |
| `process_3d_images.py` | Segments nuclei and cytoplasm for one batch of images; saves results to `all_wells_measurements.csv` |
| `functions.py` | Helper functions for segmentation — runs Cellpose inference, builds cytoplasm masks, measures intensity, density, and morphology features |
| `evaluate_segmentation.ipynb` | Benchmarks Cellpose segmentation against a hand-labeled "ground truth" image (287-tri/B02) |

### Classification

| File | Purpose |
|------|---------|
| `classify_cells.sh` | Slurm script — passes parameters from `classification_params.json` to `classification.py` |
| `classification_params.json` | Processing parameters for classification — cell types, channels, training data location |
| `classification.py` | Classifies all cells in one batch of images using a random forest model; saves results to `data.csv` |

### Misc

| File | Purpose |
|------|---------|
| `test_gpu.sh` | Quick tester script for GPU availability/usage |
| `models.py` | Wrapper around Cellpose models |
| `cellpose2.yml` | Conda environment spec for `cellpose2` *(watch yo nomenclature... tsk, tsk, Jake)* |
| `count_nuclei.py` | Similar to `process_3d_images.py`, but optimized for getting nuclei count. Uses naive thresholding for cell type |

## Getting Started

```bash
# (if running on Engaging cluster)
module load miniforge

# Build the environment
mamba env create -f cellpose2.yml
mamba activate cellpose2

# Run segmentation (example)
sbatch cellpose_gpu.sh 15-287-tri

# Run classification (example)
sbatch classify_cells.sh 15
```

## Output

- `all_wells_measurements.csv` — per-cell 3D morphology, intensity, and density features from segmentation
- `data.csv` — integrates data from all subfolder all_wells_measurements.csv + cell type label from RF

---