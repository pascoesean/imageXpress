import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from itertools import product
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import pickle

np.random.seed(42)


### CONSTANTS ###

# input features for PCA, RF, ...
features = ['nuclear_volume_voxels', 'cytoplasm_volume_voxels',
       'channel_1_nuclear_mean', 'channel_1_nuclear_max',
       'channel_1_nuclear_std', 'channel_1_cytoplasm_mean',
       'channel_1_cytoplasm_max', 'channel_1_cytoplasm_std', 'channel_1_nc_ratio',
       'channel_2_nuclear_mean', 'channel_2_nuclear_max',
       'channel_2_nuclear_std', 'channel_2_cytoplasm_mean',
       'channel_2_cytoplasm_max', 'channel_2_cytoplasm_std', 'channel_2_nc_ratio',
       'channel_3_nuclear_mean', 'channel_3_nuclear_max',
       'channel_3_nuclear_std', 'channel_3_cytoplasm_mean',
       'channel_3_cytoplasm_max', 'channel_3_cytoplasm_std', 'channel_3_nc_ratio',
       'channel_4_nuclear_mean', 'channel_4_nuclear_max',
       'channel_4_nuclear_std', 'channel_4_cytoplasm_mean',
       'channel_4_cytoplasm_max', 'channel_4_cytoplasm_std', 'channel_4_nc_ratio',
       'channel_5_nuclear_mean', 'channel_5_nuclear_max',
       'channel_5_nuclear_std', 'channel_5_cytoplasm_mean',
       'channel_5_cytoplasm_max', 'channel_5_cytoplasm_std', 'channel_5_nc_ratio',
       'centroid_z', 'axis_major_length', 'axis_minor_length', 'aspect_ratio',
       'nn_dist', 'nn5_mean_dist', 'num_neighbors_within_50_um', 'sphericity',
       ]

channels = {
    1: 'DAPI',
    2: 'Mac',
    3: 'NFkB',
    4: 'pcJun',
    5: 'actin',
}

cell_types = {
    1: 'epithelial',
    2: 'ESC',
    3: 'macrophage',
}


### FUNCTIONS ###

def plot_pca(df: pd.DataFrame, features: list, labels=None, n_components=2) -> None:
    """ Plots PCA of `df`, assumes 2 components, also shows PC histograms! """

    X = df[features].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=components, columns=[f'PC{i}' for i in range(1, n_components+1)])

    if labels is not None:
        pca_df[labels] = df[labels].to_numpy()
        sns.pairplot(pca_df, hue=labels, plot_kws={'size': 1, 'alpha': 0.2})

    else:
        sns.pairplot(pca_df, plot_kws={'size': 1, 'alpha': 0.2})


def plot_histograms(df: pd.DataFrame, channels: dict, thresholds=None, labels=None) -> None:
    """
    Handy lil function to plot channel intensity histograms, colored by `labels`
    and with `thresholds` annotated, optionally.
    """

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


def generate_cell_type_mask(df: pd.DataFrame, exp: int | str, well: str, culture_type: str, method='rf') -> None:
    """
    Given a `df` with cell type labels, create an image mask where each nucleus is labelled:
        1: epithelial
        2: ESC
        3: macrophage
    Exports the mask to /home/jdweiss1/orcd/scratch/{exp}/{culture_type}/masks
    """

    # given a df with cell type predictions, output a tiff cell type mask
    file = f'/home/jdweiss1/orcd/scratch/{str(exp)}/{culture_type}/masks/{well}_nuclear_masks.tif'
    img = tifffile.imread(file)

    type_codes = {'background': 0, 'epithelial': 1, 'ESC': 2, 'macrophage': 3, 'unknown': 4}

    # restrict df to specific donor and well
    filter = df['well_id'] == well
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
    tifffile.imwrite(f'/home/jdweiss1/orcd/scratch/{exp}/{culture_type}/masks/{well}_cell_type_{method}.tif', cell_type_mask)



### BIG functions

def load_data(exp: int | str, culture_type: str) -> pd.DataFrame:
    """
    Loads data from all_wells_measurements.csv for all donors within `exp`.
    Compiles and returns the data as one DataFrame.
    """

    # load measurement data (from cellpose ouptut)
    data = pd.read_csv(f'/home/jdweiss1/orcd/scratch/{str(exp)}/{culture_type}/all_wells_measurements.csv')

    # add column b/c was missed in first pass analysis
    data['aspect_ratio'] = data['axis_major_length'] / data['axis_minor_length']

    return data



def get_training_labels(tri_data_14_low_res: pd.DataFrame) -> pd.DataFrame:
    """
    Reads labeled tri-culture images and matches labels to `tri_data`.
    Returns a DataFrame with labeled training data from triculture.
    """

    training_data = pd.DataFrame()

    # map tri-culture cell-type labels to nuclear ids
    tri_wells = ('B06', 'B07')
    for well in tri_wells:

        file = f'/home/jdweiss1/orcd/scratch/14-low-res/tri/labels/{well}_cell_type.tif'
        labels = tifffile.imread(file)

        filter = tri_data_14_low_res['well_id'] == well
        df = tri_data_14_low_res[filter].copy()
        df['cell_type'] = None

        # build distance tree from centroid coords
        centroids = df[['centroid_z', 'centroid_y', 'centroid_x']].to_numpy() # z, y, x
        tree = KDTree(centroids)

        # transfer cell type labels from labeled image to nuclear mask
        for id, cell in cell_types.items():
            # list of coordinates for each labelled cell
            coords = np.argwhere(labels == id) # z, y, x
            if len(coords) == 0:
                continue

            # search the tree for nearest neighbors to each cell
            distances, indices = tree.query(coords)

            # filter out matches that are too far (>10 pixels)
            max_dist = 10
            valid = distances < max_dist

            # map the returned row indices back to nucleus_id
            nearest_nucleus_ids = df['nucleus_id'].to_numpy()[indices[valid]]
            df.loc[df['nucleus_id'].isin(nearest_nucleus_ids), 'cell_type'] = cell

        training_subset = df[df['cell_type'].notna()]
        training_data = pd.concat([training_data, training_subset])

    return training_data


# RF classifier

def rf_method(training_data: pd.DataFrame, data: pd.DataFrame, plot=False) -> tuple[pd.DataFrame, RandomForestClassifier]:
    """ Assigns RF cell types. Returns a modified copy of a DataFrame AND the RF model itself. """

    X = training_data[features].to_numpy()
    y = training_data['cell_type'].to_numpy()

    # now, train RF on all available training data
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced')
    rf.fit(X, y)

    # apply RF labels to whole dataset
    data['rf_cell_type'] = rf.predict(data[features])

    if plot:

        # feature importance
        importance = rf.feature_importances_
        forest_importance = pd.Series(importance, index=features).sort_values(ascending=False)[:20]

        _, ax = plt.subplots(figsize=(12,6))
        forest_importance.plot.bar(ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        plt.tight_layout()
        plt.savefig('figs_12/cell_type_rf_features.png')

    return data, rf


def eval_rf_cv(training_data: pd.DataFrame) -> dict:
    """
    - Trains on 80% of the data, validates on 20%.
    - 5-fold CV
    """

    X = training_data[features].to_numpy()
    y = training_data['cell_type'].to_numpy()

    all_y_true, all_y_pred = [], []

    for i, (train, test) in enumerate(KFold(n_splits=5).split(X)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        #print(f"\n=== Fold {i} ===")
        #print(classification_report(y_test, y_pred))
        #print(confusion_matrix(y_test, y_pred, labels=['epithelial', 'ESC', 'macrophage']))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print(f"\n=== Overall CV Perfomance ===")
    print(classification_report(all_y_true, all_y_pred))
    print(confusion_matrix(all_y_true, all_y_pred, labels=['epithelial', 'ESC', 'macrophage']))

    rf_dict = classification_report(all_y_true, all_y_pred)
    return rf_dict


# cell class probabilities

def filter_low_confidence_cells(data: pd.DataFrame, rf: RandomForestClassifier, make_mask=False) -> pd.DataFrame:

    probs = rf.predict_proba(data[features])
    max_prob = probs.max(axis=1) # max_prob = prob corresponding to selected class label

    plt.subplots()
    plt.hist(max_prob, bins=50)
    plt.savefig('figs_12/cell_type_conf_hist.png')

    # filter out cells w/ less-than-majority confidence
    low_conf_mask = max_prob < 0.5
    low_conf_idx = data['nucleus_id'][low_conf_mask]
    print("\n=== Low-Confidence Cells ===")
    print(data['rf_cell_type'][low_conf_mask].value_counts())
    data = data[~low_conf_mask]

    if make_mask:
        # make low-confidence cells mask
        img_mask = tifffile.imread('/home/jdweiss1/orcd/scratch/12/287-tri/masks/B02_nuclear_masks.tif').copy()
        img_mask[~np.isin(img_mask, low_conf_idx)] = 0
        tifffile.imwrite('/home/jdweiss1/orcd/scratch/12/287-tri/masks/B02_low_prob.tif', img_mask)

    return data





if __name__ == '__main__':

    # TRI-CULTURE

    tri_df = load_data('tri')

    plot_histograms(tri_df, channels)
    plot_pca(tri_df.dropna(), features)

    training_df = get_training_labels(tri_df)

    print('\nTraining Data:')
    print(training_df['cell_type'].value_counts())

    # plot training data in PC space
    plt.subplots()
    plot_pca(training_df.dropna(), features, labels='cell_type')
    plt.savefig('figs_12/cell_type_pca_training.png')

    # run RF classification
    tri_df, rf = rf_method(training_df, tri_df, plot=True)
    rf_dict = eval_rf_cv(training_df)
    tri_df = filter_low_confidence_cells(tri_df, rf)

    print('\n=== Tri-culture Classification ===')
    print(tri_df['rf_cell_type'].value_counts())

    # save RF model
    with open('models/rf_12.pkl', "wb") as file:
        pickle.dump(rf, file)

    # plot *all* tri-culture data in PC space
    plt.subplots()
    plot_pca(tri_df.dropna(), features, labels='rf_cell_type')
    plt.savefig('figs_12/cell_type_pca_tri_rf.png')


    # generate cell type masks for validation
    for well in ('B06', 'B07', 'B08', 'B09'):
        generate_cell_type_mask(tri_df, well, 'tri', 'rf')


    # write out csv! (all donors compiled)
    tri_df.to_csv('/home/jdweiss1/orcd/scratch/12/tri_data_iso.csv')


    # CO-CULTURE

    co_df = load_data('co')
    co_df['rf_cell_type'] = rf.predict(co_df[features])
    co_df = filter_low_confidence_cells(co_df, rf)

    print('\n=== Co-culture Classification ===')
    print(co_df['rf_cell_type'].value_counts())

    co_df.to_csv('/home/jdweiss1/orcd/scratch/12/co_data.csv')


    # MONO-CULTURE

    mono_df = load_data('mono')
    mono_df['rf_cell_type'] = rf.predict(mono_df[features])
    co_df = filter_low_confidence_cells(mono_df, rf)

    print('\n=== Mono-culture ===')
    print(mono_df['rf_cell_type'].value_counts())

    mono_df['rf_cell_type'] = 'epithelial'

    mono_df.to_csv('/home/jdweiss1/orcd/scratch/12/mono_data.csv')


    # ESC-MAC

    em_df = load_data('escmac1')
    em_df['rf_cell_type'] = rf.predict(em_df[features])
    em_df = filter_low_confidence_cells(em_df, rf)

    print('\n=== ESC-Mac 1 ===')
    print(em_df['rf_cell_type'].value_counts())

    em_df = em_df[em_df['rf_cell_type'] != 'epithelial']

    em_df.to_csv('/home/jdweiss1/orcd/scratch/12/escmac1_data_iso.csv')

    em_df = load_data('escmac2')
    em_df['rf_cell_type'] = rf.predict(em_df[features])
    em_df = filter_low_confidence_cells(em_df, rf)

    print('\n=== ESC-Mac 2 ===')
    print(em_df['rf_cell_type'].value_counts())

    em_df = em_df[em_df['rf_cell_type'] != 'epithelial']

    em_df.to_csv('/home/jdweiss1/orcd/scratch/12/escmac2_data.csv')
