# Whole-cohort model for sex prediction (no age groups)
from sklearn.metrics import pairwise
from scipy.special import logit
import scipy.io
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import time
import os
import sys

import argparse
import ast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DATA_DIR, SUBJECT_INFO_CSV, DATA_SPLITS_MAT, RESULTS_DIR


parser = argparse.ArgumentParser()
parser.add_argument('--contype', type=str, required=True, help="connectome type")
parser.add_argument('--outer_folds', type=int, default=100, help="Number of data split (default: 100)")
parser.add_argument('--inner_folds', type=int, default=5, help="cv folds in data split (default: 5)")
parser.add_argument('--train_ratio', type=float, default=0.8, help="Training data ratio (default: 0.8)")
parser.add_argument('--split0328', action='store_true', help='use split same as krakencoder fine-tuned at 0328')

args = parser.parse_args()
outer_folds = args.outer_folds
inner_folds = args.inner_folds
contype = args.contype
train_ratio = args.train_ratio
split0328 = args.split0328
seed = 42

print("START")
print(f"Parameters: contype={contype}, outer_folds={outer_folds}, inner_folds={inner_folds}")
print(f"Train ratio={train_ratio}, split0328={split0328}")


def loadmat(filename):
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


# Load demographic data
demograph = pd.read_csv(SUBJECT_INFO_CSV)
demo_age = demograph['age']
demo_sex = demograph['sex'].map({'M': 1, 'F': 0})

df_demograph = pd.DataFrame({
    'age': demo_age,
    'sex_index': demo_sex
})

print(f"Total subjects loaded: {len(df_demograph)}")
print(f"Sex distribution: M={demo_sex.sum()}, F={(demo_sex==0).sum()}")

if split0328:  # Use fine-tuned krakencoder split
    print("Using split0328 (krakencoder fine-tuned split)")
    Msplit = scipy.io.loadmat(DATA_SPLITS_MAT, simplify_cells=True)

inputfile = os.path.join(DATA_DIR, f"ls3_{contype}_1419subj.mat") 
Cdata = scipy.io.loadmat(inputfile, simplify_cells=True)['C']  # read the flavor data 
nroi = Cdata[0].shape[0]
trimask = np.triu_indices(nroi, 1) 
n_conn = len(Cdata[0][trimask])
Cflat = np.vstack([x[trimask] for x in Cdata])  # extract the upper triangle (shape: n_subjects x n_edges)

print(f"Connectivity data loaded: {Cflat.shape[0]} subjects, {n_conn} connections")

# Select data based on split type
if split0328:  # Use same datasplit as fine-tuned krakencoder
    Cflat_select = Cflat[Msplit['subjidx_test'], :]
    df_demo_select = df_demograph.iloc[Msplit['subjidx_test'], :].reset_index(drop=True)
    data_df = pd.concat([df_demo_select, pd.DataFrame(Cflat_select)], axis=1)
    print(f"Using held-out test data from split0328: {len(data_df)} subjects")
else:
    data_df = pd.concat([df_demograph, pd.DataFrame(Cflat)], axis=1)
    print(f"Using all subjects: {len(data_df)} subjects")

# Initialize results storage
tr_acc = np.empty((outer_folds, 1))
te_acc = np.empty((outer_folds, 1))         
test_pred = []
best_params_list = []

# Main cross-validation loop
for outer_iter in range(outer_folds):
    print(f"Outer fold {outer_iter + 1}/{outer_folds}")
    rng = seed + outer_iter
    np.random.seed(rng)

    # Split by sex to ensure balanced representation
    males = data_df[data_df['sex_index'] == 1]
    females = data_df[data_df['sex_index'] == 0]

    # Calculate test size to maintain train_ratio
    test_size_m = int(len(males) * (1 - train_ratio))
    test_size_f = int(len(females) * (1 - train_ratio))

    # Split males and females separately
    m_tr, m_te = train_test_split(males, test_size=test_size_m, random_state=rng) 
    f_tr, f_te = train_test_split(females, test_size=test_size_f, random_state=rng)

    # Combine and shuffle
    train_df = pd.concat([m_tr, f_tr]).sample(frac=1, random_state=rng) 
    test_df = pd.concat([m_te, f_te]).sample(frac=1, random_state=rng)

    print(f"  Train: {len(train_df)} subjects ({len(m_tr)}M, {len(f_tr)}F)")
    print(f"  Test: {len(test_df)} subjects ({len(m_te)}M, {len(f_te)}F)")

    # Prepare features and labels
    X_train = train_df.iloc[:, 2:]  # Skip 'age' and 'sex_index' columns
    X_test = test_df.iloc[:, 2:]
    y_train = train_df['sex_index']
    y_test = test_df['sex_index']

    # Grid search with stratified CV
    param_grid = {'C': [0.001, 0.01, 1, 10, 100, 1000]}
    model = LogisticRegression(penalty='l2', max_iter=1000, random_state=rng)

    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=rng)   
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params_list.append(grid_search.best_params_)

    # Compute accuracies
    tr_acc[outer_iter, :] = accuracy_score(y_train, best_model.predict(X_train))
    y_pred = best_model.predict(X_test)
    te_acc[outer_iter, :] = accuracy_score(y_test, y_pred)
    test_pred.append([y_test.values, y_pred])



# Save results
result = {
    'train_accuracy': tr_acc,
    'test_accuracy': te_acc,
    'test_result': test_pred,
    'best_params': best_params_list
}

# Save results
output_dir = os.path.join(RESULTS_DIR, 'wholecohort/logistic_regression_ensemble')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    f"{contype}_wholecohort_stratifiedCV_split0328_{split0328}_train{int(train_ratio*100)}.mat"
)

scipy.io.savemat(output_file, result)

## Paths used in this script (from config.py):
## - DATA_DIR = "/rawdata_dynseed"
## - SUBJECT_INFO_CSV = "/ls3_subjinfo_agesexcog_details_1419subj.csv"
## - DATA_SPLITS_MAT = "/ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat"
## - RESULTS_DIR = "/home/out_log/results_HCPdata_Kraken/wholecohort/logistic_regression_ensemble"
