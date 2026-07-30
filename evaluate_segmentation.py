import tifffile
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import label

from functions import load_channel_stack


# LABELS

labels_file = '/home/jdweiss1/orcd/scratch/15/287-tri-B02/labels/B02_nuclear.tif'
labels = tifffile.imread(labels_file)

# multi-class --> binary classification
labels_2d = (labels > 0)
labels = labels_2d.flatten().astype(int)


# DATA

measurements = pd.read_csv('../15/287-tri-B02/all_wells_measurements.csv')

measurements['centroid_x'] = measurements['centroid_x_um'] / 0.6793
measurements['centroid_y'] = measurements['centroid_y_um'] / 0.6793
measurements['centroid_z'] = measurements['centroid_z_um'] / 5

def update_df(df, model, preds):

    iou       = metrics.jaccard_score(labels, preds)
    f1        = metrics.f1_score(labels, preds)
    mcc       = metrics.matthews_corrcoef(labels, preds)
    precision = metrics.precision_score(labels, preds)
    recall    = metrics.recall_score(labels, preds)

    new_row = pd.DataFrame({'IOU': iou, 'F1': f1, 'MCC': mcc, 'Precision': precision, 'Recall': recall}, index=[model])

    return pd.concat([df, new_row])


def confusion_matrix(model, preds):

    display = metrics.ConfusionMatrixDisplay.from_predictions(
        labels, preds, cmap=plt.cm.Blues
    )
    display.ax_.set_title(f'{model} Confusion Matrix')
    plt.savefig(f'figs/{model}_cm.png')
    plt.show()


def curves(model, preds):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # precision-recall curve
    _ = metrics.PrecisionRecallDisplay.from_predictions(
        labels, preds, name=model, plot_chance_level=True, despine=True, ax=ax1
    )
    ax1.set_title('Precision-Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')

    # ROC curve
    _ = metrics.RocCurveDisplay.from_predictions(
        labels, preds, name=model, plot_chance_level=True, despine=True, ax=ax2
    )
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')

    fig.suptitle('Cellpose2 Performance on 287-tri/B02')
    plt.tight_layout()
    plt.show()


def get_error_map(preds_2d, filepath):
    error_map = np.zeros((*preds_2d.shape, 3))

    tp =  labels_2d  &  preds_2d
    fp = ~labels_2d  &  preds_2d
    fn =  labels_2d  & ~preds_2d

    error_map = np.zeros((*preds_2d.shape, 3), dtype=np.float32)
    error_map[tp] = [0, 1, 0]   # TP -- green
    error_map[fp] = [1, 0, 0]   # FP -- red
    error_map[fn] = [0, 0, 1]   # FN -- blue

    tifffile.imwrite(filepath, error_map)



# PREDICTIONS

df = pd.DataFrame(columns=['IOU', 'F1', 'MCC', 'Precision', 'Recall'])

all_preds = []

for cellprob in (-1, 0, 1, 2):

    preds_file = f'/home/jdweiss1/orcd/scratch/15/287-tri-B02/masks/B02_nuclear_masks_cellpose2_cellprob{cellprob}.tif'
    preds = tifffile.imread(preds_file)

    # take the max intensity projection of the predicted nuclear masks
    preds = preds.max(axis=0)

    # multi-class --> binary classification
    preds_2d = (preds > 0)
    preds = preds_2d.flatten().astype(int)
    all_preds.append(preds)

    df = update_df(df, f'Cellpose2_{cellprob}', preds)

# cellprob1 has highest F1

preds_file = f'/home/jdweiss1/orcd/scratch/15/287-tri-B02/masks/B02_nuclear_masks_cellpose2_cellprob1.tif'
preds = tifffile.imread(preds_file)

# take the max intensity projection of the predicted nuclear masks
preds = preds.max(axis=0)

# multi-class --> binary classification
preds_2d = (preds > 0)
preds = preds_2d.flatten().astype(int)
all_preds.append(preds)

confusion_matrix('Cellpose2', all_preds[1])

curves('Cellpose2_1', preds)

get_error_map(preds_2d, 'figs/error_map.tif')


# split image into smaller regions, evaluate performance, plot performance vs cell density in region

x, y = preds_2d.shape # (2048, 2048)
grid = 8
patch_size = x // grid # 512
xy_pixel_um = 0.6793

data = {
    'IOU': [],
    'F1': [],
    'density': [],
    'EpCAM': [],
    'DAPI': [],
}

for i in range(grid):

    for j in range(grid):

        lower_x = i*patch_size
        upper_x = lower_x + patch_size

        lower_y = j*patch_size
        upper_y = lower_y + patch_size

        mini_mask_2d = preds_2d[lower_x:upper_x, lower_y:upper_y]
        mini_mask = (mini_mask_2d > 0).flatten().astype(int)

        mini_labels_2d = labels_2d[lower_x:upper_x, lower_y:upper_y]
        mini_labels = (mini_labels_2d > 0).flatten().astype(int)

        iou = metrics.jaccard_score(mini_labels, mini_mask)
        f1 = metrics.f1_score(mini_labels, mini_mask)

        # NOTE: cell counts are calculated from segmentation output, not hand-labeled data. it's way easier this way,
        #       but there's potential information leakage. what if the segmentation is under-classifying, affecting
        #       both the F1/IOU score *and* the density measurement?
        roi_mask = ((measurements['centroid_x'] >= lower_x) & (measurements['centroid_x'] < upper_x) &
                    (measurements['centroid_y'] >= lower_y) & (measurements['centroid_y'] < upper_y))

        num_nuclei = roi_mask.sum()
        region_area = (patch_size * xy_pixel_um) ** 2
        region_density = num_nuclei / region_area

        epcam = measurements['channel_5_nuclear_max'][roi_mask].sum()
        dapi = measurements['channel_1_nuclear_max'][roi_mask].sum()

        data['IOU'].append(iou)
        data['F1'].append(f1)
        data['density'].append(region_density)
        data['EpCAM'].append(epcam)
        data['DAPI'].append(dapi)


density_df = pd.DataFrame(data, index=list(range(grid ** 2)))
density_df = density_df[density_df['density'] != 0]

_, axs = plt.subplots(1, 2, figsize=(10,4))

sns.regplot(density_df, x='EpCAM', y='IOU', ax=axs[0])
axs[0].set_xlabel('Local Density (um^-2)')
axs[0].set_ylabel('IOU')

sns.regplot(density_df, x='EpCAM', y='F1', ax=axs[1])
axs[1].set_xlabel('Local Density (um^-2)')
axs[1].set_ylabel('F1')

plt.savefig('figs/recall_vs_density.png')


