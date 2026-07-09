import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import pickle

np.random.seed(42)


# load parameters from json

EXP_ID = sys.argv[1]

with open('classification_params.json') as f:
    all_params = json.load(f)

if EXP_ID not in all_params:
    raise KeyError(f"Exp ID '{EXP_ID}' not found in classification_params.json")

def integerize(dict):
    """
    Converts a dict with str keys to int keys.
    """
    return {int(k): v for k, v in dict.items()}

p = all_params[EXP_ID]
CHANNELS = integerize(p['channels'])
CELL_TYPES = integerize(p['cell_types'])
SUBFOLDERS = p['subfolders']
LABEL_IMGS = p['label_imgs']
OUTPUT_DIR = p['output_dir']


# input features for PCA, RF, ...
FEATURES = ['nuclear_volume_voxels', 'cytoplasm_volume_voxels',
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


def generate_cell_type_mask(df: pd.DataFrame, well: str, subfolder: str) -> None:
    """
    Given a `df` with cell type labels, create an image mask where each nucleus is labelled:
        1: epithelial
        2: ESC
        3: macrophage
    Exports the mask to /home/jdweiss1/orcd/scratch/{exp}/{culture_type}/masks
    """

    # given a df with cell type predictions, output a tiff cell type mask
    file = f'/home/jdweiss1/orcd/scratch/{EXP_ID}/{subfolder}/masks/{well}_nuclear_masks.tif'
    img = tifffile.imread(file)

    type_codes = CELL_TYPES.copy()
    type_codes[0] = 'background'

    # restrict df to specific donor and well
    filter = df['well_id'] == well
    df = df[filter]

    # build a lookup array for fast indexing
    lookup = np.zeros(img.max() + 1, dtype=np.uint8)
    for _, row in df.iterrows():
        lookup[int(row['nucleus_id'])] = type_codes[row['rf_cell_type']]

    # map nuclear ids to cell types
    cell_type_mask = lookup[img]
    tifffile.imwrite(f'/home/jdweiss1/orcd/scratch/{EXP_ID}/{subfolder}/masks/{well}_cell_type.tif', cell_type_mask)



### BIG functions

def load_data() -> pd.DataFrame:
    """
    Loads data from all_wells_measurements.csv for all subfolders within the experiment.
    Compiles and returns the data as one DataFrame.
    """

    dfs = []

    for subfolder in SUBFOLDERS:
        # load measurement data (from cellpose ouptut)
        file = f'/home/jdweiss1/orcd/scratch/{EXP_ID}/{subfolder}/all_wells_measurements.csv'
        print(f'Loading data from {file}')
        data = pd.read_csv(file)
        data['subfolder'] = subfolder

        # add columns that were missed in first pass analysis
        data['aspect_ratio'] = data['axis_major_length'] / data['axis_minor_length']
        for channel in range(1, 6):
            data[f'channel_{channel}_nc_ratio'] = (
                data[f'channel_{channel}_nuclear_integrated'] / (data[f'channel_{channel}_cytoplasm_integrated'] + 1e-9)
            )

        dfs.append(data)

    return pd.concat(dfs)



def get_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads labeled training images and matches labels to `df`.
    Returns a DataFrame with labeled training data.
    """

    training_data = pd.DataFrame()

    # map tri-culture cell-type labels to nuclear ids
    for file in LABEL_IMGS:
        print(f'\nLoading training data from {file}')
        labels = tifffile.imread(file)
        well = re.search(r'([A-Z](?:0[2-9]|1[01]))', file).group(1)
        subfolder = re.search(r'([^/]+)/labels', file).group(1)

        filter = (df['well_id'] == well) & (df['subfolder'] == subfolder)
        df = df[filter]
        df['cell_type'] = None

        # build distance tree from centroid coords
        centroids = df[['centroid_z', 'centroid_y', 'centroid_x']].to_numpy() # z, y, x
        tree = KDTree(centroids)

        # transfer cell type labels from labeled image to nuclear mask
        for id, cell in CELL_TYPES.items():
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
            filter = df['nucleus_id'].isin(nearest_nucleus_ids)
            df['cell_type'][filter] = cell

        training_subset = df[df['cell_type'].notna()]
        training_data = pd.concat([training_data, training_subset])

    print(f'\n=== Training data ===')
    print(training_data['cell_type'].value_counts())

    return training_data


# RF classifier

def load_rf_model(training_data: pd.DataFrame, features: list) -> RandomForestClassifier:
    """
    If it exists, loads RF model from pickle. Otherwise, trains an RF on training_data[features].
    """

    pickle_path = Path(OUTPUT_DIR + f'rf_{EXP_ID}.pkl')

    if pickle_path.is_file():
        with open(pickle_path, 'rb') as file:
            print(f'\nLoading RF pickle from {pickle_path}')
            rf = pickle.load(file)
    else:
        print('\nTraining RF model')
        X = training_data[features].to_numpy()
        y = training_data['cell_type'].to_numpy()
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced')
        rf.fit(X, y)
        with open(pickle_path, 'wb') as file:
            pickle.dump(rf, file)

    return rf



def plot_rf_features(rf: RandomForestClassifier, features: list[str]) -> None:
    """
    Plots feature importances (mean decrease gini) from rf.
    """
    importance = rf.feature_importances_
    forest_importance = pd.Series(importance, index=features).sort_values(ascending=False)[:20]

    _, ax = plt.subplots(figsize=(12,6))
    forest_importance.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    plt.tight_layout()
    plt.savefig(f'figs_{EXP_ID}/cell_type_rf_features.png')


def eval_rf_cv(training_data: pd.DataFrame, features: list[str], k=5) -> dict:
    """
    - Trains on 80% of the data, validates on 20%.
    - k-fold CV
    """

    X = training_data[features].to_numpy()
    y = training_data['cell_type'].to_numpy()

    all_y_true, all_y_pred = [], []

    for train, test in KFold(n_splits=k).split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print(f"\n=== Overall CV Perfomance ===")
    print(classification_report(all_y_true, all_y_pred))
    print(confusion_matrix(all_y_true, all_y_pred, labels=['epithelial', 'ESC', 'macrophage']))

    rf_dict = classification_report(all_y_true, all_y_pred)
    return rf_dict


# cell class probabilities

def filter_low_confidence_cells(data: pd.DataFrame, features: list[str], rf: RandomForestClassifier, make_mask=False) -> pd.DataFrame:

    probs = rf.predict_proba(data[features])
    max_prob = probs.max(axis=1) # max_prob = prob corresponding to selected class label

    plt.subplots()
    plt.hist(max_prob, bins=50)
    plt.savefig(f'figs_{EXP_ID}/cell_type_conf_hist.png')

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

    df = load_data()
    training_data = get_training_labels(df)
    classifier = load_rf_model(training_data, FEATURES)
    plot_rf_features(classifier, FEATURES)
    eval_rf_cv(training_data, FEATURES)
    df['rf_cell_type'] = classifier.predict(df[FEATURES])
    df.to_csv(OUTPUT_DIR + 'data.csv')
