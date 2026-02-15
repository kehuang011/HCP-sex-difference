#if the result does not contain the seed number, it is seed 42

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.special import logit
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import pairwise, balanced_accuracy_score, make_scorer, recall_score
from sklearn.svm import SVC
import os
import sys

import argparse
import ast


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, SUBJECT_INFO_CSV, DATA_SPLITS_MAT, RESULTS_DIR


start_time = time.time()
parser = argparse.ArgumentParser(description="Parse input parameters for nested cross-validation.")

parser.add_argument('--bins', nargs='+', type=int, required=True, help="List of bins, e.g., [8, 15, 22, 29, 36, 50, 65, 80, 100]")
parser.add_argument('--dataname', type=str, required=True, help="Name of the dataset (input & save), e.g., 'krakenLS3dynseed_1419subj_20250304_132140_ep000500_encoded' ")
parser.add_argument('--param_grid', type=str, default="{'C': [0.001, 0.01, 1, 10, 100, 1000]}", help="Parameter grid for cv")
parser.add_argument('--outer_folds', type=int, default=100, help="Number of data split")
parser.add_argument('--inner_folds', type=int, default=5, help="cv folds in data split")
parser.add_argument('--downsample_set', type=str, default='test', help='downsample by train set or test set (when using train_val data as training set), e.g., "train" or "test" or "train_test" ')
parser.add_argument('--downsample', action='store_true', help='downsample')
parser.add_argument('--add_trainingYA', type=str, help='add original/new YA predicted from krakencoder as training data / no_addtraining')
parser.add_argument('--split0328', action='store_true', help='use split same as krakencoder fine-tuned at 0328 (403 training)')
parser.add_argument('--testsize', type=int, default=20, help="# of F/M in test data; 20 (10 F/M); 30 (15 F/M)")
parser.add_argument('--add_originalYA_per_sex', type=int, default=10, help="# of added original YA data; 10 (10 F/M); 15 (15 F/M)")
parser.add_argument('--train_downsample_size', type=int, default=15, help="# downsample  F/M (default: 15 F/ 15M)")
parser.add_argument('--method', type=str, help='logistic / krr (cosine) to predict sex based on krakencoder low-dimesional data')

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

def balanced_accuracy_with_all_classes(y_true, y_predict):
    # union of labels in true and predicted (site labels)
    all_classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_predict)]))
    return recall_score(y_true, y_predict, labels=all_classes, average="macro", zero_division=0)
custom_bal_acc = make_scorer(balanced_accuracy_with_all_classes)

bins = args.bins
outer_folds = args.outer_folds
inner_folds = args.inner_folds
dataname = args.dataname
seed = 42 # 1234
add_trainingYA = args.add_trainingYA
downsample = args.downsample
split0328 = args.split0328
testsize = args.testsize
add_originalYA_per_sex = args.add_originalYA_per_sex
train_downsample_size = args.train_downsample_size
method = args.method

kr_data = loadmat(os.path.join(DATA_DIR, f'{dataname}.mat'))
demograph = pd.read_csv(SUBJECT_INFO_CSV)

kr_age = demograph['age']
kr_sex = demograph['sex'].map({'M': 1, 'F': 0})
kr_site = demograph['Site']
print(kr_sex)

df_demograph = pd.DataFrame({
    'age': kr_age,
    'sex_index': kr_sex,
    'site': kr_site})
df_demograph['age_category'] = pd.cut(df_demograph['age'], bins=bins, right=False)

dim_atlas = 128 
n_connectome = 128

result_types = {}

if add_trainingYA == 'no_addtraining':
    add_originalYA_per_sex = 0

if (method == 'logistic') or (method == 'SVC_cosine') or (method == 'SVC') or (method == 'SVC_linear'):
    param_grid = {'C': [0.001, 0.01, 1, 10, 100, 1000]}
elif method == 'krr':
    param_grid = {'alpha': [0.001, 0.01, 1, 10, 100, 1000]}

for type in ['fusion', 'fusionFC', 'fusionSC']:
    #if (dataname == 'krakenLS3dyn403_1419subj_20250327_152011_ep000500_encoded') or (dataname == 'krakenLS3dyn403demofine_sex.w100_20250506_222320_ep000500_fusion_encoded') or (dataname == 'krakenLS3dyn403demofine_age0sexsite.w100_20250825_145054_ep000500_encoded') or (dataname == 'krakenLS3dyn403demofine_agesexsite.w100_20250508_161030_ep000500_encoded'): 
    #     ######## new krakencoder 0328 or sex 0308 or sex0328 with site effects removed; new YA
    if (dataname == 'krakenLS3dynICV403demofineSEX100_20251107_144040_ep000500_encoded') or (dataname == 'krakenLS3dynICV403demofineAGE0SEXSITEcond100_20251107_010029_ep000500_encoded'): 
        # corrected coco439; SC ICV: sex-finetuned, sex-site-finetuned 
        if split0328 == True: # only use fine-tuned test data
            Msplit = scipy.io.loadmat(DATA_SPLITS_MAT, simplify_cells=True)
            data = kr_data['predicted_alltypes'][type]['encoded']
            data_select = data[Msplit['subjidx_test'],:]; df_demograph_select = df_demograph.iloc[Msplit['subjidx_test'],:].reset_index(drop=True)
            data_df = pd.concat([df_demograph_select, pd.DataFrame(data_select)], axis=1)
        else:    
            data = kr_data['predicted_alltypes'][type]['encoded']
            data_df = pd.concat([df_demograph, pd.DataFrame(data)], axis=1)
            
        if (add_trainingYA == 'newYA_train') or (add_trainingYA == 'both'):         
            d_trkr = data[Msplit['subjidx_train'],:]
            df_demo_trkr = df_demograph.iloc[Msplit['subjidx_train'],:].reset_index(drop=True)
            d_valkr = data[Msplit['subjidx_val'],:]
            df_demo_valkr = df_demograph.iloc[Msplit['subjidx_val'],:].reset_index(drop=True)
            df_trkr = pd.concat([df_demo_trkr, pd.DataFrame(d_trkr)], axis=1) 
            df_valkr = pd.concat([df_demo_valkr, pd.DataFrame(d_valkr)], axis=1)
            df_trkr = pd.concat([df_trkr, df_valkr])
            grouped_trkr = df_trkr.groupby('age_category')
            age_category_trainkr_dict = {age_category: group for age_category, group in grouped_trkr}                      

    #elif dataname == 'ls3_adapt1419_dynseed_20240413_210723_ep002000_encoded': ####### original krakencoder (not fine-tuned); new YA
    elif dataname =='ls3_adapt403_dynseedICV_20240413_210723_ep002000_encoded': ####### original krakencoder corrected coco439, SC icv (not fine-tuned)
        if split0328 == True: # only use fine-tuned test data
            Msplit = scipy.io.loadmat(DATA_SPLITS_MAT, simplify_cells=True)
            data = kr_data['predicted_alltypes'][type]['encoded']
            data_select = data[Msplit['subjidx_test'],:]; df_demograph_select = df_demograph.iloc[Msplit['subjidx_test'],:].reset_index(drop=True)
            data_df = pd.concat([df_demograph_select, pd.DataFrame(data_select)], axis=1)
        else:    
            data = kr_data['predicted_alltypes'][type]['encoded']
            data_df = pd.concat([df_demograph, pd.DataFrame(data)], axis=1)

        if add_trainingYA == 'newYA_train':
            d_trkr = data[Msplit['subjidx_train'],:]
            df_demo_trkr = df_demograph.iloc[Msplit['subjidx_train'],:].reset_index(drop=True)
            d_valkr = data[Msplit['subjidx_val'],:]
            df_demo_valkr = df_demograph.iloc[Msplit['subjidx_val'],:].reset_index(drop=True)
            df_trkr = pd.concat([df_demo_trkr, pd.DataFrame(d_trkr)], axis=1) 
            df_valkr = pd.concat([df_demo_valkr, pd.DataFrame(d_valkr)], axis=1)
            df_trkr = pd.concat([df_trkr, df_valkr])
            grouped_trkr = df_trkr.groupby('age_category')
            age_category_trainkr_dict = {age_category: group for age_category, group in grouped_trkr}  

    else: # other datasets
        data = kr_data['predicted_alltypes'][type]['encoded']
        n_allsample = len(data)
        print(f"bin: {bins}")
        data_df = pd.concat([df_demograph, pd.DataFrame(data)], axis=1)


    grouped = data_df.groupby('age_category')
    age_category_dict = {age_category: group for age_category, group in grouped}

    print(f"param_grid: {param_grid}")
    data_dict = {} 

    for i in range(len(bins)-1):
        np.random.seed(seed + i)
        df_age_range = age_category_dict[pd.Interval(left=bins[i], right=bins[i+1], closed='left')]
        print(f'{bins[i]}-{bins[i+1]}')
        best_params_list = []

        logi_coeff = np.empty((outer_folds, n_connectome))
        haufe_coeff = np.empty((outer_folds, n_connectome))
        tr_acc = np.empty((outer_folds, 1))
        te_acc = np.empty((outer_folds, 1))
        te_pred = []

        for outer_iter in range(outer_folds):
            rng = seed + outer_iter
            np.random.seed(rng)
            if split0328 == True: # use split0328 (split of fine-tuned krakencoder 0328)  
                males = df_age_range[df_age_range['sex_index'] == 1]
                females = df_age_range[df_age_range['sex_index'] == 0]
                m_tr, m_te = train_test_split(males, test_size = int(testsize/2), random_state = rng) 
                f_tr, f_te = train_test_split(females, test_size = int(testsize/2), random_state = rng)

                if add_trainingYA == 'newYA_train': # or (i==4) and (add_trainingYA == 'newYA_train'): #########
                    age_trkr = age_category_trainkr_dict[pd.Interval(left=bins[i], right=bins[i+1], closed='left')]
                    m_kr = age_trkr[age_trkr['sex_index'] == 1]
                    f_kr = age_trkr[age_trkr['sex_index'] == 0]
                    tr_data = pd.concat([m_tr, f_tr, m_kr, f_kr]).sample(frac=1, random_state = rng) 
                    te_data = pd.concat([m_te, f_te]).sample(frac=1, random_state = rng)
                    add_originalYA_per_sex = 0
                else:
                    tr_data = pd.concat([m_tr, f_tr]).sample(frac=1, random_state = rng) 
                    te_data = pd.concat([m_te, f_te]).sample(frac=1, random_state = rng)
                    
                if downsample == True:
                    tr_data_f = tr_data[tr_data['sex_index'] == 0].sample(n = train_downsample_size, random_state = rng)
                    tr_data_m = tr_data[tr_data['sex_index'] == 1].sample(n = train_downsample_size, random_state = rng)
                    tr_data = pd.concat([tr_data_f, tr_data_m]).sample(frac=1, random_state = rng)

            tr_data = tr_data.reset_index(drop=True)
            te_data = te_data.reset_index(drop=True)
            combined_data = pd.concat([tr_data, te_data]).reset_index(drop=True)
            combined_data['site'], site_cat = pd.factorize(combined_data['site'], sort=True)
            tr_data = combined_data.iloc[:len(tr_data), :].reset_index(drop=True)
            te_data = combined_data.iloc[len(tr_data):, :].reset_index(drop=True)

            X_tr = tr_data.iloc[:, 4:]; y_train = tr_data['site']
            X_te = te_data.iloc[:, 4:]; y_test = te_data['site']
            print(X_tr.shape, X_te.shape)

            sex_train = tr_data['sex_index']
            inner_cv = StratifiedKFold(n_splits = inner_folds, shuffle=True, random_state = rng)
            inner_splits = list(inner_cv.split(X_tr, sex_train))

            if method == 'logistic':
                model = LogisticRegression(multi_class='multinomial', penalty='l2', max_iter=1000, random_state = rng)
                grid_search = GridSearchCV(model, param_grid, cv=inner_splits, scoring=custom_bal_acc, n_jobs=-1)
                grid_search.fit(X_tr, y_train)
                best_model = grid_search.best_estimator_
                tr_acc[outer_iter] = best_model.score(X_tr, y_train)
                y_pred = best_model.predict(X_te)

            elif method == 'krr':
                model = RidgeClassifier(random_state = rng)
                grid_search = GridSearchCV(model, param_grid, cv=inner_splits, scoring=custom_bal_acc, n_jobs=1)
                X_tr_cosine = pairwise.cosine_similarity(X_tr)
                X_te_cosine = pairwise.cosine_similarity(X_te, X_tr)
                grid_search.fit(X_tr_cosine, y_train)
                best_model = grid_search.best_estimator_
                tr_acc[outer_iter] = best_model.score(X_tr_cosine, y_train)
                y_pred = best_model.predict(X_te_cosine)

            elif method == 'SVC_cosine':    
                model = SVC(kernel = 'precomputed')
                grid_search = GridSearchCV(model, param_grid, cv=inner_splits, scoring=custom_bal_acc, n_jobs=1)
                X_tr_cosine = pairwise.cosine_similarity(X_tr)
                X_te_cosine = pairwise.cosine_similarity(X_te, X_tr)
                grid_search.fit(X_tr_cosine, y_train)
                best_model = grid_search.best_estimator_
                tr_acc[outer_iter] = best_model.score(X_tr_cosine, y_train)
                y_pred = best_model.predict(X_te_cosine)

            elif method == 'SVC':
                model = SVC()
                grid_search = GridSearchCV(model, param_grid, cv=inner_splits, scoring=custom_bal_acc, n_jobs=1)
                grid_search.fit(X_tr, y_train)
                best_model = grid_search.best_estimator_
                tr_acc[outer_iter] = best_model.score(X_tr, y_train)
                y_pred = best_model.predict(X_te)

            elif method == 'SVC_linear':
                model = SVC(kernel = 'linear')
                grid_search = GridSearchCV(model, param_grid, cv=inner_splits, scoring=custom_bal_acc, n_jobs=1)
                grid_search.fit(X_tr, y_train)
                best_model = grid_search.best_estimator_
                tr_acc[outer_iter] = best_model.score(X_tr, y_train)
                y_pred = best_model.predict(X_te)

            all_classes = np.unique(np.concatenate([np.unique(y_test), np.unique(y_pred)]))
            te_acc[outer_iter, :] = recall_score(y_test, y_pred, labels=all_classes, average="macro", zero_division=0)
            te_pred.append([y_test, y_pred])
    
        result = {
            'original_coefficient': logi_coeff,
            'haufe_coeff': haufe_coeff,
            'train_accuracy': tr_acc,
            'test_accuracy': te_acc,
            'test_result': te_pred
        }

        data_dict[str(bins[i])+'-'+str(bins[i+1])] = result

        if downsample == True:
            # Determine subdirectory based on dataname
            if dataname == 'ls3_adapt403_dynseedICV_20240413_210723_ep002000_encoded':
                subdir = 'krakencoder_original'
            elif dataname == 'krakenLS3dyn403_1419subj_20250327_152011_ep000500_encoded': 
                subdir = 'krakencoder500_0328'
            elif dataname == 'krakenLS3dynICV403demofineSEX100_20251107_144040_ep000500_encoded': 
                subdir = 'krakencoder500_sex0508'
            elif dataname == 'krakenLS3dynICV403demofineAGE0SEXSITEcond100_20251107_010029_ep000500_encoded':
                subdir = 'krakencoder500_sex0508_site_rm'
            
            output_dir = os.path.join(RESULTS_DIR, subdir)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'results1110/encoded_pred{type}{bins}_stratifiedCV_split0328_test{testsize}_add{add_trainingYA}oriYA{add_originalYA_per_sex}_downsampledtrain{train_downsample_size*2}_{method}_site.mat')
            scipy.io.savemat(output_file, data_dict)

## Paths used in this script (from config.py):
## - DATA_DIR = "/LS3"
## - SUBJECT_INFO_CSV = "/ls3_subjinfo_agesexcog_details_1419subj.csv"
## - DATA_SPLITS_MAT = "/ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat"
## - RESULTS_DIR = "/home/out_log/results_HCPdata_Kraken/newYA95/"