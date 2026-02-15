# new HCPYA data; add training dataor not add
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

# Try to import local config, fall back to defaults if not found
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import DATA_DIR, SUBJECT_INFO_CSV, DATA_SPLITS_MAT, RESULTS_DIR
except ImportError:
    print("WARNING: config.py not found. Using default paths.")
    print("Please copy config_template.py to config.py and set your paths.")
    DATA_DIR = "/midtier/cocolab/scratch/kwj2001/LS3/rawdata_dynseed"
    SUBJECT_INFO_CSV = "/midtier/cocolab/scratch/kwj2001/LS3/ls3_subjinfo_agesexcog_details_1419subj.csv"
    DATA_SPLITS_MAT = "/midtier/cocolab/scratch/kwj2001/LS3/ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat"
    RESULTS_DIR = "./outputs/results"

parser = argparse.ArgumentParser()
parser.add_argument('--bins', nargs='+', type=int, required=True, help="List of bins, e.g., [8, 15, 22, 29, 36, 50, 65, 80, 100]")
parser.add_argument('--contype', type=str, required=True, help="connectome type")
parser.add_argument('--outer_folds', type=int, default=100, help="Number of data split")
parser.add_argument('--inner_folds', type=int, default=5, help="cv folds in data split")
parser.add_argument('--downsample', action='store_true', help='downsample')
parser.add_argument('--add_trainingYA', type=str, help='add original/new YA predicted from krakencoder as training data / no_addtraining')
parser.add_argument('--split0328', action='store_true', help='use split same as krakencoder fine-tuned at 0328')
parser.add_argument('--testsize', type=int, default=20, help="# of F/M in test data; 20 (10 F/M); 30 (15 F/M)")
parser.add_argument('--train_downsample_size', type=int, default=15, help="# downsample  F/M (default: 15 F/ 15M)")

args = parser.parse_args()
bins = args.bins
outer_folds = args.outer_folds
inner_folds = args.inner_folds
contype = args.contype
seed = 42
add_trainingYA = args.add_trainingYA
downsample = args.downsample
split0328 = args.split0328
testsize = args.testsize
train_downsample_size = args.train_downsample_size


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


demograph = pd.read_csv(SUBJECT_INFO_CSV)
demo_age = demograph['age']
demo_sex = demograph['sex'].map({'M': 1, 'F': 0})

df_demograph = pd.DataFrame({
    'age': demo_age,
    'sex_index': demo_sex})


df_demograph['age_category'] = pd.cut(df_demograph['age'], bins=bins, right=False)

if split0328 == True: ## (newYA: use fine-tuned krakencoder / compare the performance with fine-tuned 0328)
    Msplit = scipy.io.loadmat(DATA_SPLITS_MAT, simplify_cells=True)

inputfile = os.path.join(DATA_DIR, f"ls3_{contype}_1419subj.mat") 
Cdata=scipy.io.loadmat(inputfile,simplify_cells=True)['C'] #read the flavor data 
nroi=Cdata[0].shape[0]
trimask=np.triu_indices(nroi,1) 
n_conn = len(Cdata[0][trimask])
Cflat=np.vstack([x[trimask] for x in Cdata]) # extract the upper triangle (shape: n_subjects x n_edges)


if split0328: # use same datasplit as fine-tuned krakencoder
    Cflat_select = Cflat[Msplit['subjidx_test'],:]
    df_demo_select = df_demograph.iloc[Msplit['subjidx_test'],:].reset_index(drop=True) # select the fine-tuned 0328 held out data to use
    data_df = pd.concat([df_demo_select, pd.DataFrame(Cflat_select)], axis=1)

    if add_trainingYA == 'newYA_train':    # select training data used to fine-tuned krakencoder (kr0328 or sex) 
        Cflats_trkr = Cflat[Msplit['subjidx_train'],:]
        df_demo_trkr = df_demograph.iloc[Msplit['subjidx_train'],:].reset_index(drop=True)
        Cflats_valkr = Cflat[Msplit['subjidx_val'],:]
        df_demo_valkr = df_demograph.iloc[Msplit['subjidx_val'],:].reset_index(drop=True)
        Cflats_df_trkr = pd.concat([df_demo_trkr, pd.DataFrame(Cflats_trkr)], axis=1) 
        Cflats_df_valkr = pd.concat([df_demo_valkr, pd.DataFrame(Cflats_valkr)], axis=1)
        Cflats_df_trkr = pd.concat([Cflats_df_trkr, Cflats_df_valkr])
        grouped_tr_kr = Cflats_df_trkr.groupby('age_category', observed=False)
        Cflats_trcontype = {age_category: group for age_category, group in grouped_tr_kr}  
elif not split0328:
    data_df = pd.concat([df_demograph, pd.DataFrame(Cflat)], axis=1)       
grouped = data_df.groupby('age_category', observed=False)
Cflats_contype = {age_category: group for age_category, group in grouped}       

data_dict = {} 
for i in range(len(bins)-1):
    print(f'{bins[i]}-{bins[i+1]}')
    best_params_list = []

    coeff = np.empty((outer_folds, n_conn))
    haufe_coeff = np.empty((outer_folds, n_conn))
    tr_acc = np.empty((outer_folds, 1))
    te_acc = np.empty((outer_folds, 1))         
    test_pred = []
    interval = pd.Interval(left=bins[i], right=bins[i+1], closed='left')

    for outer_iter in range(outer_folds):
        print(f"BIN {bins[i]}-{bins[i+1]}: Outer fold {outer_iter}")
        rng = seed + outer_iter
        np.random.seed(rng)

        grp = Cflats_contype[interval]
        males = grp[grp['sex_index'] == 1]
        females = grp[grp['sex_index'] == 0]

        m_tr, m_te = train_test_split(males, test_size = int(testsize/2), random_state = rng) 
        f_tr, f_te = train_test_split(females, test_size = int(testsize/2), random_state = rng)

        if add_trainingYA == 'newYA_train':  # add training data from fine-tuned krakencoder
            kr_grp = Cflats_trcontype[interval]                    
            m_kr = kr_grp[kr_grp['sex_index'] == 1]
            f_kr = kr_grp[kr_grp['sex_index'] == 0]
            train_df = pd.concat([m_tr, f_tr, m_kr, f_kr]).sample(frac=1, random_state = rng)  
        else:
            train_df = pd.concat([m_tr, f_tr]).sample(frac=1, random_state = rng) 
        test_df = pd.concat([m_te, f_te]).sample(frac=1, random_state = rng)

        if downsample:
            subs_f = train_df[train_df['sex_index'] == 0].sample(n = train_downsample_size, random_state = rng)
            subs_m = train_df[train_df['sex_index'] == 1].sample(n = train_downsample_size, random_state = rng)
            train_df = pd.concat([subs_f, subs_m]).sample(frac=1, random_state = rng)

        X_train = train_df.iloc[:, 3:]
        X_test = test_df.iloc[:, 3:]
        y_train = train_df['sex_index']
        y_test = test_df['sex_index']
        param_grid = {'C': [0.001, 0.01, 1, 10, 100, 1000]}
        model = LogisticRegression(penalty='l2', max_iter=1000, random_state = rng)

        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state = rng)   
        grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1) # change it to -1 to enable multiple cpus
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        cov_matrix = np.cov(X_train, rowvar=False)         # haufe transformed coefficients
        w_ridge = best_model.coef_.flatten()
        haufe_beta = cov_matrix @ w_ridge
        del cov_matrix
        coeff[outer_iter, :] = w_ridge
        haufe_coeff[outer_iter, :] = haufe_beta 

        tr_acc[outer_iter, :] = accuracy_score(y_train, best_model.predict(X_train))
        y_pred = best_model.predict(X_test)
        te_acc[outer_iter, :] = accuracy_score(y_test, y_pred)
        test_pred.append([y_test, y_pred])

    result = {
        'original_coefficient': coeff,
        'haufe_coeff': haufe_coeff,
        'train_accuracy': tr_acc,
        'test_accuracy': te_acc,
        'test_result': test_pred,
    }

    data_dict[str(bins[i])+'-'+str(bins[i+1])] = result

output_dir = os.path.join(RESULTS_DIR, 'logistic_regression_ensemble')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'{contype}{bins}stratifiedCV_split0328_test{testsize}{add_trainingYA}_downsampledtrain{train_downsample_size*2}.mat')
scipy.io.savemat(output_file, data_dict)

## Paths used in this script (from config.py):
## - DATA_DIR = "/rawdata_dynseed"
## - SUBJECT_INFO_CSV = "/ls3_subjinfo_agesexcog_details_1419subj.csv"
## - DATA_SPLITS_MAT = "/ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat"
## - RESULTS_DIR = "/home/out_log/results_HCPdata_Kraken/newYA95/logistic_regression_ensemble"

