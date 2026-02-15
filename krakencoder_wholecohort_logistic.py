# Whole-cohort krakencoder model for sex prediction (no age groups)
# Only uses logistic regression

import scipy.io
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import time
import argparse
import os
import sys
from datetime import datetime


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, SUBJECT_INFO_CSV, DATA_SPLITS_MAT, RESULTS_DIR


start_time = time.time()

parser = argparse.ArgumentParser(description="Whole-cohort sex prediction from krakencoder latent space")

parser.add_argument('--dataname', type=str, required=True, 
                    help="Dataset name: 'krakenLS3dynICV403demofineSEX100_20251107_144040_ep000500_encoded' or 'ls3_adapt403_dynseedICV_20240413_210723_ep002000_encoded'")
parser.add_argument('--outer_folds', type=int, default=100, help="Number of data splits (default: 100)")
parser.add_argument('--inner_folds', type=int, default=5, help="CV folds for hyperparameter tuning (default: 5)")
parser.add_argument('--train_ratio', type=float, default=0.8, help="Training data ratio (default: 0.8)")
parser.add_argument('--split0328', action='store_true', help='Use split same as krakencoder fine-tuned at 0328 (403 training)')

args = parser.parse_args()

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


# Parse arguments
dataname = args.dataname
outer_folds = args.outer_folds
inner_folds = args.inner_folds
train_ratio = args.train_ratio
split0328 = args.split0328
seed = 42

# Validate dataname
allowed_datasets = [
    'krakenLS3dynICV403demofineSEX100_20251107_144040_ep000500_encoded',
    'ls3_adapt403_dynseedICV_20240413_210723_ep002000_encoded'
]

if dataname not in allowed_datasets:
    raise ValueError(f"dataname must be one of: {allowed_datasets}")

# Load krakencoder data
kr_data = loadmat(os.path.join(DATA_DIR, f'{dataname}.mat'))
demograph = pd.read_csv(SUBJECT_INFO_CSV)

# Prepare demographics
kr_age = demograph['age']
kr_sex = demograph['sex'].map({'M': 1, 'F': 0})

df_demograph = pd.DataFrame({
    'age': kr_age,
    'sex_index': kr_sex
})

print(f"Total subjects loaded: {len(df_demograph)}")
print(f"Sex distribution: M={kr_sex.sum()}, F={(kr_sex==0).sum()}")

# Parameters
dim_atlas = 128 
n_connectome = 128
param_grid = {'C': [0.001, 0.01, 1, 10, 100, 1000]}

print(f"Parameter grid: {param_grid}")

# Process each fusion type
result_types = {}

for type in ['fusion', 'fusionFC', 'fusionSC']:
    print(f"Processing type: {type}")
    
    # Load and prepare data based on dataset
    if dataname == 'krakenLS3dynICV403demofineSEX100_20251107_144040_ep000500_encoded':
        # Sex fine-tuned model
        if split0328:
            # Use only held-out test data
            Msplit = scipy.io.loadmat(DATA_SPLITS_MAT, simplify_cells=True)
            data = kr_data['predicted_alltypes'][type]['encoded']
            data_select = data[Msplit['subjidx_test'], :]
            df_demo_select = df_demograph.iloc[Msplit['subjidx_test'], :].reset_index(drop=True)
            data_df = pd.concat([df_demo_select, pd.DataFrame(data_select)], axis=1)
            print(f"Using held-out test data from split0328: {len(data_df)} subjects")
        else:
            # Use all subjects
            data = kr_data['predicted_alltypes'][type]['encoded']
            data_df = pd.concat([df_demograph, pd.DataFrame(data)], axis=1)
            
    elif dataname == 'ls3_adapt403_dynseedICV_20240413_210723_ep002000_encoded':
        # Original krakencoder (not fine-tuned)
        if split0328:
            # Use only held-out test data
            Msplit = scipy.io.loadmat(DATA_SPLITS_MAT, simplify_cells=True)
            data = kr_data['predicted_alltypes'][type]['encoded']
            data_select = data[Msplit['subjidx_test'], :]
            df_demo_select = df_demograph.iloc[Msplit['subjidx_test'], :].reset_index(drop=True)
            data_df = pd.concat([df_demo_select, pd.DataFrame(data_select)], axis=1)
            print(f"Using held-out test data from split0328: {len(data_df)} subjects")
        else:
            # Use all subjects
            data = kr_data['predicted_alltypes'][type]['encoded']
            data_df = pd.concat([df_demograph, pd.DataFrame(data)], axis=1)

    # Initialize results storage
    print("Starting cross-validation loop...")
    tr_acc = np.empty((outer_folds, 1))
    te_acc = np.empty((outer_folds, 1))
    te_pred = []
    best_params_list = []

    # Main cross-validation loop (whole cohort)
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
        tr_data = pd.concat([m_tr, f_tr]).sample(frac=1, random_state=rng)
        te_data = pd.concat([m_te, f_te]).sample(frac=1, random_state=rng)

        print(f"  Train: {len(tr_data)} subjects ({len(m_tr)}M, {len(f_tr)}F)")
        print(f"  Test: {len(te_data)} subjects ({len(m_te)}M, {len(f_te)}F)")

        # Prepare features and labels (skip age and sex_index columns)
        X_tr = tr_data.iloc[:, 2:]
        y_train = tr_data['sex_index']
        X_te = te_data.iloc[:, 2:]
        y_test = te_data['sex_index']

        # Logistic regression with grid search
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=rng)
        model = LogisticRegression(penalty='l2', max_iter=1000, random_state=rng)
        grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_tr, y_train)
        best_model = grid_search.best_estimator_
        best_params_list.append(grid_search.best_params_)

        # Compute accuracies
        tr_acc[outer_iter] = best_model.score(X_tr, y_train)
        y_pred = best_model.predict(X_te)
        te_acc[outer_iter, :] = accuracy_score(y_test, y_pred)
        te_pred.append([y_test.values, y_pred])

    # Store results for this type
    result = {
        'train_accuracy': tr_acc,
        'test_accuracy': te_acc,
        'test_result': te_pred,
        'best_params': best_params_list
    }

    result_types[type] = result

# Determine output filename based on dataset
if dataname == 'krakenLS3dynICV403demofineSEX100_20251107_144040_ep000500_encoded':
    subdir = 'sex_finetuned'
elif dataname == 'ls3_adapt403_dynseedICV_20240413_210723_ep002000_encoded':
    subdir = 'original'


output_dir = os.path.join(RESULTS_DIR, subdir)
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    f"wholecohort_encoded_pred_split0328_{split0328}_train{int(train_ratio*100)}_logistic.mat"
)

scipy.io.savemat(output_file, result_types)

## Paths used in this script (from config.py):
## - DATA_DIR = "/LS3"
## - SUBJECT_INFO_CSV = "/ls3_subjinfo_agesexcog_details_1419subj.csv"
## - DATA_SPLITS_MAT = "/ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat"
## - RESULTS_DIR = "/home/out_log/results_HCPdata_Kraken/wholecohort/krakencoder/"



