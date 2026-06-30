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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
import pickle

np.random.seed(42)


### CONSTANTS ###

# if True, evaluate and compare RF to thresholding method
# if False, only run RF method
EVAL = False

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

donors = ('284', '287', '304')



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


def find_gmm_threshold(values: pd.Series, n_components=2) -> np.ndarray:
    """
    Fits a Gaussian Mixture Model to `values` (a Series of intensity values)
    to identify two populations. Returns a threshold, which is the midpoint
    of the estimated population means.
    """

    # convert to log scale since intensity is log-distributed
    vals = np.log1p(values).to_numpy().reshape(-1, 1)

    # fit n_components model to distinguish between distributions of cells
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(vals)
    means = sorted(gmm.means_.flatten())

    # find midpoint between means, convert from log to linear scale
    threshold_log = np.mean(means)
    return np.expm1(threshold_log)


def classify_by_threshold(df: pd.DataFrame, thresholds: list) -> np.ndarray:
    """
    Given a list of `thresholds` for the cell trace and EpCAM channels,
    assign each cell to a cell type based on its cell trace and EpCAM intensity.
    """

    # assign each cell to a class based on Mac, EpCAM intensity rel. to thresholds
    mac_thresh, epcam_thresh = thresholds
    mac_high   =  df['channel_4_nuclear_mean']  >  mac_thresh
    epcam_high =  df['channel_5_nuclear_mean']  >  epcam_thresh

    conditions = [
        epcam_high  & ~mac_high,
        ~epcam_high & ~mac_high,
        ~epcam_high & mac_high,
        epcam_high  & mac_high
    ]
    choices = ['epithelial', 'ESC', 'macrophage', 'doublet']

    # map each cell to a choice based on mac_high and epcam_high condition
    return np.select(conditions, choices, default='unknown')


def generate_cell_type_mask(df: pd.DataFrame, donor: int | str, well: str, exp: str, method='rf') -> None:
    """
    Given a `df` with cell type labels, create an image mask where each nucleus is labelled:
        1: epithelial
        2: ESC
        3: macrophage
    Exports the mask to /home/jdweiss1/orcd/scratch/15/{donor}-{exp}/masks
    """

    # given a df with cell type predictions, output a tiff cell type mask
    file = f'/home/jdweiss1/orcd/scratch/15/{str(donor)}-{exp}/masks/{well}_nuclear_masks_cellpose2.tif'
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
    tifffile.imwrite(f'/home/jdweiss1/orcd/scratch/15/{str(donor)}-{exp}/masks/{well}_cell_type_{method}.tif', cell_type_mask)



### BIG functions

def load_data(exp: str) -> pd.DataFrame:
    """
    Loads data from all_wells_measurements_cellpose2.csv for all donors within `exp`.
    Compiles and returns the data as one DataFrame.
    """

    # load measurement data (from cellpose ouptut)
    data = pd.DataFrame()

    for donor in donors:
        mini_df = pd.read_csv(f'/home/jdweiss1/orcd/scratch/15/{donor}-{exp}/all_wells_measurements_cellpose2.csv')
        mini_df['donor_id'] = int(donor)
        data = pd.concat([data, mini_df])

    # add column b/c was missed in first pass analysis
    data['aspect_ratio'] = data['axis_major_length'] / data['axis_minor_length']

    # filter out high intensity Mac cell trace
    filter = (data['channel_4_nuclear_mean'] < 1e4)
    data = data[filter]

    return data



def get_training_labels(tri_data: pd.DataFrame) -> pd.DataFrame:
    """
    Reads labeled tri-culture images and matches labels to `tri_data`.
    Returns a DataFrame with labeled training data from triculture.
    """

    training_data = pd.DataFrame()

    # map tri-culture cell-type labels to nuclear ids
    tri_wells = ('B02', 'B03')
    for donor, well in product(donors, tri_wells):

        file = f'/home/jdweiss1/orcd/scratch/15/{donor}-tri/labels/{well}_cell_type.tif'
        labels = tifffile.imread(file)

        filter = (tri_data['donor_id'] == int(donor)) & (tri_data['well_id'] == well)
        df = tri_data[filter].copy()
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
        training_subset['culture_type'] = 'tri'
        training_data = pd.concat([training_data, training_subset])

    return training_data


# thresholding method

def thresholds_method(data: pd.DataFrame, plot=False) -> tuple[pd.DataFrame, list]:
    """ Assigns threshold cell types. Returns a modified copy of a DataFrame AND a list of threshold cutoffs. """

    # fit GMM on *all* donors, *all* wells data for Mac and EpCAM channels
    mac_threshold = find_gmm_threshold(data['channel_4_nuclear_mean'])
    eeo_threshold = find_gmm_threshold(data['channel_5_nuclear_mean'])

    channels = {4: 'Mac cell trace', 5: 'EpCAM'}
    thresholds = [mac_threshold, eeo_threshold]

    if plot:

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
        plt.axvline(x=np.log1p(mac_threshold), color='k', ls='--')
        plt.axhline(y=np.log1p(eeo_threshold), color='k', ls='--')
        plt.savefig('figs/cell_type_hist2d.png')

    # ASSIGN LABELS!!!

    # first, bulk all data together and do 1 GMM
    data['thresh_cell_type'] = classify_by_threshold(data, thresholds)

    #print(data['thresh_cell_type'].value_counts())

    filter = data['thresh_cell_type'] != 'doublet'
    data = data[filter]

    return data, thresholds



def eval_thresholds(training_data: pd.DataFrame, thresholds: list) -> dict:
    """ Returns a dictionary containing the classification report (recall, precision, F1, ...) """

    # apply thresholds to the hand-labeled subset
    labeled = training_data.copy()
    labeled = classify_by_threshold(labeled, thresholds)

    # exclude doublets from evaluation
    eval_df = labeled[labeled['thresh_cell_type'] != 'doublet']
    print(f"\n=== Thresholding: ===")
    print(classification_report(eval_df['cell_type'], eval_df['thresh_cell_type']))
    print(confusion_matrix(eval_df['cell_type'], eval_df['thresh_cell_type'],
                        labels=['epithelial', 'ESC', 'macrophage']))

    thresh_dict = classification_report(
        eval_df['cell_type'],
        eval_df['thresh_cell_type'],
        output_dict=True
    )

    return thresh_dict


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
        plt.savefig('figs/cell_type_rf_features.png')

    return data, rf



def eval_rf(training_data: pd.DataFrame) -> dict:
    """
    - Trains on 2 donors (tri-culture & mono-culture)
    - Evaluates on the 3rd donor (tri-culture ONLY)
    - Does this 3 times (3-fold cross validation)
    """

    X = training_data[features].to_numpy()
    y = training_data['cell_type'].to_numpy()

    groups = training_data['donor_id'].to_numpy()
    logo = LeaveOneGroupOut()

    all_y_true = []
    all_y_pred = []

    # train on two donors and validate on the third
    for donor, (train_idx, test_idx) in zip(np.unique(groups), logo.split(training_data, groups=groups)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # train `balanced` so that we don't bias toward a more prevalent class (epithelial)
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        print(f"\n=== Held-out donor: {donor} ===")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred, labels=["epithelial", "ESC", "macrophage"]))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print("\n=== Overall Leave-One-Donor-Out Performance ===")
    print(classification_report(all_y_true, all_y_pred))
    print(confusion_matrix(all_y_true, all_y_pred, labels=["epithelial", "ESC", "macrophage"]))

    rf_dict = classification_report(all_y_true, all_y_pred, output_dict=True)

    return rf_dict



### compare RF and threshold performance

def compare_rf_and_thresh(thresh_dict: dict, rf_dict: dict) -> None:

    thresh_df = pd.DataFrame(thresh_dict).T
    rf_df = pd.DataFrame(rf_dict).T

    cell_types = ['epithelial', 'ESC', 'macrophage']

    # Long-format dataframe for seaborn
    plot_df = pd.concat([
        pd.DataFrame({
            'cell_type': cell_types,
            'method': 'Thresholding',
            'f1_score': thresh_df.loc[cell_types, 'f1-score'].values
        }),
        pd.DataFrame({
            'cell_type': cell_types,
            'method': 'Random Forest',
            'f1_score': rf_df.loc[cell_types, 'f1-score'].values
        })
    ])

    # Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=plot_df,
        x='cell_type',
        y='f1_score',
        hue='method'
    )

    plt.ylabel('F1')
    plt.title('Classification Performance by Cell Type')
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig('figs/cell_type_method_f1s.png')



# cell class probabilities

def filter_low_confidence_cells(data: pd.DataFrame, rf: RandomForestClassifier, make_mask=False) -> pd.DataFrame:

    probs = rf.predict_proba(data[features])
    max_prob = probs.max(axis=1) # max_prob = prob corresponding to selected class label

    plt.subplots()
    plt.hist(max_prob, bins=50)
    plt.savefig('figs/cell_type_conf_hist.png')

    # filter out cells w/ less-than-majority confidence
    low_conf_mask = max_prob < 0.5
    low_conf_idx = data['nucleus_id'][low_conf_mask]
    print(data['rf_cell_type'][low_conf_mask].value_counts())
    data = data[~low_conf_mask]

    if make_mask:
        # make low-confidence cells mask
        img_mask = tifffile.imread('/home/jdweiss1/orcd/scratch/15/287-tri/masks/B02_nuclear_masks_cellpose2.tif').copy()
        img_mask[~np.isin(img_mask, low_conf_idx)] = 0
        tifffile.imwrite('/home/jdweiss1/orcd/scratch/15/287-tri/masks/B02_low_prob.tif', img_mask)

    return data





if __name__ == '__main__':

    # TRI-CULTURE

    tri_df = load_data('tri')

    training_df = get_training_labels(tri_df)

    print('\nTraining Data:')
    print(training_df['cell_type'].value_counts())

    # plot training data in PC space
    plt.subplots()
    plot_pca(training_df.dropna(), features, labels='cell_type')
    plt.savefig('figs/cell_type_pca_training.png')

    # run RF classification
    tri_df, rf = rf_method(training_df, tri_df, plot=True)
    rf_dict = eval_rf(training_df)
    tri_df = filter_low_confidence_cells(tri_df, rf)

    print('\nTri-culture Classification:')
    print(tri_df['rf_cell_type'].value_counts())

    # save RF model
    with open('models/tri-rf.pkl', "wb") as file:
        pickle.dump(rf, file)

    # plot *all* tri-culture data in PC space
    plt.subplots()
    plot_pca(tri_df.dropna(), features, labels='rf_cell_type')
    plt.savefig('figs/cell_type_pca_tri_rf.png')


    # generate cell type masks for validation
    for donor, well in product((284, 287, 304), ('B02', 'B03', 'B04', 'B05')):
        generate_cell_type_mask(tri_df, donor, well, 'tri', 'rf')


    if EVAL:
        # run thresholds classification
        tri_df, thresholds = thresholds_method(tri_df)
        thresh_dict = eval_thresholds(training_df, thresholds)

        # generate cell type masks for validation
        for well in ('B02', 'B03', 'B04', 'B05'):
            generate_cell_type_mask(tri_df, 287, well, 'tri', 'thresh')

        compare_rf_and_thresh(thresh_dict, rf_dict)

    # write out csv! (all donors compiled)
    #tri_df.to_csv('/home/jdweiss1/orcd/scratch/15/tri_data.csv')



    # CO-CULTURE

    co_df = load_data('co')

    co_df['rf_cell_type'] = rf.predict(co_df[features])
    co_df = filter_low_confidence_cells(co_df, rf)

    print('\nCo-culture Classification:')
    print(co_df['rf_cell_type'].value_counts())

    # plot *all* co-culture data in PC space
    plt.subplots()
    plot_pca(co_df.dropna(), features, labels='rf_cell_type')
    plt.savefig('figs/cell_type_pca_co_rf.png')

    # generate cell type masks for validation
    for donor, well in product((284, 287, 304), ('B02', 'B03')):
        generate_cell_type_mask(co_df, donor, well, 'co', 'rf')

    # write out csv! (all donors compiled)
    #co_df.to_csv('/home/jdweiss1/orcd/scratch/15/co_data.csv')



    # MONO-CULTURE... this should be pretty easy lolz

    mono_df = load_data('mono')

    # just out of curiosity, let's see how well RF predicts mono-culture (should all be epithelial)
    mono_df['rf_cell_type'] = rf.predict(mono_df[features])
    print('\nMono-culture Classification:')
    print(mono_df['rf_cell_type'].value_counts())

    generate_cell_type_mask(mono_df, 284, 'B08', 'mono', 'rf')
    generate_cell_type_mask(mono_df, 284, 'B09', 'mono', 'rf')
    generate_cell_type_mask(mono_df, 287, 'B08', 'mono', 'rf')
    generate_cell_type_mask(mono_df, 287, 'B09', 'mono', 'rf')

    mono_df['rf_cell_type'] = 'epithelial'

    # write out csv! (all donors compiled)
    #mono_df.to_csv('/home/jdweiss1/orcd/scratch/15/mono_data.csv')
