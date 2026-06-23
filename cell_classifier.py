import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from itertools import product
from scipy.spatial import KDTree
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix

from functions import load_channel_stack



### functions ###


def plot_pca(df, features, labels=None, n_components=2):
    X = df[features].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=components, columns=[f'PC{i}' for i in range(1, n_components+1)])

    if labels is not None:
        pca_df[labels] = df[labels].to_numpy()
        sns.pairplot(pca_df, hue=labels, plot_kws={'size': 1, 'alpha': 0.5})

    else:
        sns.pairplot(pca_df, plot_kws={'size': 1, 'alpha': 0.5})


def plot_histograms(df, channels, thresholds=None, labels=None):
    _, axs = plt.subplots(1, len(channels), figsize=(14, 6))

    for i, (channel, name) in enumerate(channels.items()):
        # plot histogram (label by color optional)
        if labels is not None:
            sns.histplot(df, x=f'channel_{channel}_nuclear_mean', hue=labels, ax=axs[i])
        else:
            sns.histplot(df, x=f'channel_{channel}_nuclear_mean', ax=axs[i])

        axs[i].set_xlabel(f'mean {name} intensity')

        # plot threshold as vertical line
        if thresholds is not None:
            axs[i].axvline(x=thresholds[i], color='k', ls='--')


def find_gmm_threshold(values, n_components=2):
    # convert to log scale since intensity is log-distributed
    vals = np.log1p(values).to_numpy().reshape(-1, 1)

    # fit n_components model to distinguish between distributions of cells
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(vals)
    means = sorted(gmm.means_.flatten())

    # find midpoint between means, convert from log to linear scale
    threshold_log = np.mean(means)
    return np.expm1(threshold_log)


def classify_by_threshold(df, thresholds):
    # assign each cell to a class based on Mac, EpCAM intensity rel. to thresholds
    mac_thresh, epcam_thresh = thresholds
    mac_high   =  df['channel_4_nuclear_mean']  >  mac_thresh
    epcam_high =  df['channel_5_nuclear_mean']  >  epcam_thresh

    conditions = [
        mac_high  & ~epcam_high,
        ~mac_high & epcam_high,
        ~mac_high & ~epcam_high,
        mac_high  & epcam_high,
    ]
    choices = ['macrophage', 'epithelial', 'ESC', 'doublet']

    df['thresh_cell_type'] = np.select(conditions, choices, default='unknown')
    return df


def generate_cell_type_mask(df, donor, well, method):
    # given a df with cell type predictions, output a tiff cell type mask
    file = f'/home/jdweiss1/orcd/scratch/15/{str(donor)}-tri/masks/{well}_nuclear_masks_cellpose2.tif'
    img = tifffile.imread(file)

    type_codes = {'background': 0, 'epithelial': 1, 'ESC': 2, 'macrophage': 3, 'unknown': 4}

    # restrict df to specific donor and well
    filter = (df['donor_id'] == donor) & (df['well_id'] == well)
    df = df[filter]

    # choose the appropriate cell type label acc. to method
    if method == 'rf':
        label_col = 'rf_cell_type'
    elif method == 'thresh':
        label_col = 'thresh_cell_type'
    else:
        raise ValueError('method must be one of (rf, thresh)')

    # build a lookup array for fast indexing
    lookup = np.zeros(img.max() + 1, dtype=np.uint8)
    for _, row in df.iterrows():
        lookup[int(row['nucleus_id'])] = type_codes[row[label_col]]

    # map nuclear ids to cell types
    cell_type_mask = lookup[img]
    tifffile.imwrite(f'/home/jdweiss1/orcd/scratch/15/{str(donor)}-tri/masks/{well}_cell_type_{method}.tif', cell_type_mask)




# input features for PCA, RF, ...
features = ['nuclear_volume_voxels', 'nuclear_volume_um3',
       'cytoplasm_volume_voxels', 'cytoplasm_volume_um3',
       'channel_1_nuclear_mean', 'channel_1_nuclear_max',
       'channel_1_nuclear_std', 'channel_1_nuclear_integrated',
       'channel_1_cytoplasm_mean', 'channel_1_cytoplasm_max',
       'channel_1_cytoplasm_std', 'channel_1_cytoplasm_integrated',
       'channel_1_nc_ratio', 'channel_2_nuclear_mean', 'channel_2_nuclear_max',
       'channel_2_nuclear_std', 'channel_2_nuclear_integrated',
       'channel_2_cytoplasm_mean', 'channel_2_cytoplasm_max',
       'channel_2_cytoplasm_std', 'channel_2_cytoplasm_integrated',
       'channel_2_nc_ratio', 'channel_3_nuclear_mean', 'channel_3_nuclear_max',
       'channel_3_nuclear_std', 'channel_3_nuclear_integrated',
       'channel_3_cytoplasm_mean', 'channel_3_cytoplasm_max',
       'channel_3_cytoplasm_std', 'channel_3_cytoplasm_integrated',
       'channel_3_nc_ratio', 'channel_4_nuclear_mean', 'channel_4_nuclear_max',
       'channel_4_nuclear_std', 'channel_4_nuclear_integrated',
       'channel_4_cytoplasm_mean', 'channel_4_cytoplasm_max',
       'channel_4_cytoplasm_std', 'channel_4_cytoplasm_integrated',
       'channel_4_nc_ratio', 'channel_5_nuclear_mean', 'channel_5_nuclear_max',
       'channel_5_nuclear_std', 'channel_5_nuclear_integrated',
       'channel_5_cytoplasm_mean', 'channel_5_cytoplasm_max',
       'channel_5_cytoplasm_std', 'channel_5_cytoplasm_integrated',
       'channel_5_nc_ratio', 'centroid_x', 'centroid_y', 'centroid_z',
       'axis_major_length', 'axis_minor_length', 'aspect_ratio',
       'nn_dist', 'nn_nucleus_id', 'nn5_mean_dist',
       'num_neighbors_within_50_um', 'sphericity',
       ]

channels = {
    1: 'DAPI',
    2: 'pcJun',
    3: 'ki67',
    4: 'Mac',
    5: 'EpCAM',
}

cell_types = {
    1: 'epithelial',
    2: 'ESC',
    3: 'macrophage',
}




### import data ###

# load measurement data (from cellpose ouptut)
data = pd.DataFrame()

donors = ('284', '287', '304')

for donor in donors:
    mini_df = pd.read_csv(f'/home/jdweiss1/orcd/scratch/15/{donor}-tri/all_wells_measurements_cellpose2.csv')
    mini_df['donor_id'] = int(donor)
    data = pd.concat([data, mini_df])

# add column b/c was missed in first pass analysis
data['aspect_ratio'] = data['axis_major_length'] / data['axis_minor_length']


# load training data labels (hand-labeled)
training_data = pd.DataFrame()

wells = ('B02', 'B03')

for donor, well in product(donors, wells):
    file = f'/home/jdweiss1/orcd/scratch/15/{donor}-tri/labels/{well}_cell_type.tif'
    labels = tifffile.imread(file)

#    n_epithelial = int((labels == 1).sum())
#    n_esc        = int((labels == 2).sum())
#    n_macrophage = int((labels == 3).sum())
#    print(f'{n_epithelial=}, {n_esc=}, {n_macrophage=}')

    filter = (data['donor_id'] == int(donor)) & (data['well_id'] == well)
    df = data[filter].copy()
    df['cell_type'] = None

    # build distance tree from centroid coords
    centroids = df[['centroid_z', 'centroid_y', 'centroid_x']].to_numpy() # z, y, x
    tree = KDTree(centroids)

    for id, cell in cell_types.items():
        # list of coordinates for each labelled cell
        coords = np.argwhere(labels == id) # z, y, x
        if len(coords) == 0:
            continue

        # search the tree for nearest neighbors to each cell
        distances, indices = tree.query(coords)

        # filter out matches that are too far
        max_dist = 10
        valid = distances < max_dist

        # map the returned row indices back to nucleus_id
        nearest_nucleus_ids = df['nucleus_id'].to_numpy()[indices[valid]]
        df.loc[df['nucleus_id'].isin(nearest_nucleus_ids), 'cell_type'] = cell

    training_subset = df[df['cell_type'].notna()]
    training_data = pd.concat([training_data, training_subset])

print('\nTraining Data:')
print(training_data['cell_type'].value_counts())



### manual thresholding ###

# filter out high intensity Mac cell trace
filter = (data['channel_4_nuclear_mean'] < 1e4)
data = data[filter]

# fit GMM on *all* donors, *all* wells data for Mac and EpCAM channels
channels = {4: 'Macrophage cell trace', 5: 'EpCAM'}
mac_threshold = find_gmm_threshold(data['channel_4_nuclear_mean'])
eeo_threshold = find_gmm_threshold(data['channel_5_nuclear_mean'])
thresholds = [mac_threshold, eeo_threshold]

# 1D histograms for Mac and EpCAM channels
plt.subplots()
plot_histograms(data, channels, thresholds)
plt.savefig('figs/cell_type_hist.png')

# 2D histogram for Mac and EpCAM channels
plt.subplots()
mac = np.log1p(data['channel_4_nuclear_mean'])
epcam = np.log1p(data['channel_5_nuclear_mean'])
plt.hist2d(mac, epcam, bins=50)
plt.xlabel('Mac cell trace')
plt.ylabel('EpCAM')
plt.axvline(x=np.log1p(thresholds[0]), color='k', ls='--')
plt.axhline(y=np.log1p(thresholds[1]), color='k', ls='--')
plt.savefig('figs/cell_type_hist2d.png')

# ASSIGN LABELS!!!

# first, bulk all data together and do 1 GMM
df = classify_by_threshold(data, thresholds)

#print(df['thresh_cell_type'].value_counts())

filter = df['thresh_cell_type'] != 'doublet'
df = df[filter]



# plot training data in PC space
plt.subplots()
plot_pca(training_data.dropna(), features, labels='cell_type')
plt.savefig('figs/cell_type_pca_training.png')

# plot *all* data in PC space (bulk)
plt.subplots()
plot_pca(df.dropna(), features, labels='thresh_cell_type')
plt.savefig('figs/cell_type_pca_alldata_thresh.png')




### evaluate classification performance ###

# apply thresholds to the hand-labeled subset
labeled = training_data.copy()
labeled = classify_by_threshold(labeled, thresholds)

# exclude doublets from evaluation
eval_df = labeled[labeled['cell_type'] != 'doublet']
print('\nThresholding:')
print(classification_report(eval_df['cell_type'], eval_df['thresh_cell_type']))
print(confusion_matrix(eval_df['cell_type'], eval_df['thresh_cell_type'],
                       labels=['macrophage', 'epithelial', 'ESC']))



### generate cell type masks for validation

generate_cell_type_mask(df, 287, 'B04', 'thresh')





# RF classifier

# first, 3-fold cross validate by training on two donors and validating on the third

X = training_data[features].to_numpy()
y = training_data['cell_type'].to_numpy()
groups = training_data['donor_id'].to_numpy()

logo = LeaveOneGroupOut()

all_y_true = []
all_y_pred = []

for donor, (train_idx, test_idx) in zip(np.unique(groups), logo.split(X, y, groups)):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(f"\n=== Held-out donor: {donor} ===")
    print(classification_report(y_test, y_pred))

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=["macrophage", "epithelial", "ESC"]
        )
    )

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

print("\n=== Overall Leave-One-Donor-Out Performance ===")
print(classification_report(all_y_true, all_y_pred))

print(
    confusion_matrix(
        all_y_true,
        all_y_pred,
        labels=["macrophage", "epithelial", "ESC"]
    )
)



# now, train RF on all available training data
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced')
rf.fit(X, y)

# apply to whole dataset
df['rf_cell_type'] = rf.predict(df[features])

# plot *all* data in PC space (bulk)
plt.subplots()
plot_pca(df.dropna(), features, labels='rf_cell_type')
plt.savefig('figs/cell_type_pca_alldata_rf.png')

# feature importance
importance = rf.feature_importances_
forest_importance = pd.Series(importance, index=features).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(16,6))
forest_importance.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
plt.tight_layout()
plt.savefig('figs/cell_type_rf_features.png')



### generate cell type masks for validation

generate_cell_type_mask(df, 287, 'B04', 'rf')
generate_cell_type_mask(df, 287, 'B05', 'rf')
