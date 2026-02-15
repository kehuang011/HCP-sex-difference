import scipy.io
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from krakencoder.adaptermodel import KrakenAdapter
from krakencoder.model import Krakencoder
import argparse
import ast
import time
import os
from datetime import datetime

from krakencoder.data import (
    generate_adapt_transformer,
    load_transformers_from_file)


start_time = time.time()
parser = argparse.ArgumentParser(description="Parse input parameters for nested cross-validation.")
parser.add_argument('--param_grid', type=str, default="{'C': [0.001, 0.01, 1, 10, 100, 1000]}", help="Parameter grid for cv")
parser.add_argument('--outer_folds', type=int, default=100, help="Number of data split")
parser.add_argument('--inner_folds', type=int, default=5, help="cv folds in data split")
parser.add_argument('--is_meanfit_shift', action='store_true')
parser.add_argument('--bins', nargs='+', type=int, required=True, help="List of bins, e.g., [8, 15, 22, 29, 36, 50, 65, 80, 100]")
parser.add_argument('--split0328', action='store_true', help='use split same as krakencoder fine-tuned at 0328 (403 training)')
parser.add_argument('--add_trainingYA', type=str, help='add original/new YA predicted from krakencoder as training data / no_addtraining')
parser.add_argument('--train_downsample_size', type=int, default=15, help="# downsample  F/M (default: 15 F/ 15M)")
parser.add_argument('--testsize', type=int, default=20, help="# of F/M in test data; 20 (10 F/M); 30 (15 F/M)")
parser.add_argument('--downsample', action='store_true', help='downsample')
parser.add_argument('--model_kraken', type=str, default='sex_Krakencoder', help='or sex_site_Krakencoder')

args = parser.parse_args()

## some arguments
do_exclude_mask = False
do_include_mask = True
replace_with = 'mean'
split0328 = args.split0328
add_trainingYA = args.add_trainingYA
train_downsample_size = args.train_downsample_size
testsize = args.testsize
downsample = args.downsample
model_kraken = args.model_kraken

bins = args.bins
param_grid = ast.literal_eval(args.param_grid)
outer_folds = args.outer_folds
inner_folds = args.inner_folds
is_meanfit_shift = args.is_meanfit_shift
#data_use = args.data
seed = 42
dim_atlas = 128 

#### load data function #########
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

#### functions for generating krakencoder latent features #####
def get_conngroup_dict(conntypes):
    conngroups={}
    for c in conntypes:
        if 'fusion' in c or 'burst' in c:
            conngroups[c]=''
        elif 'FC' in c:
            conngroups[c]='FC'
        else:
            conngroups[c]='SC'
    return conngroups

def combine_input_groups(data_dict, group_dict, group_to_combine='all', combine_type='mean', normalize=False):
    """
    Combine data from different connectivity types (flavors) into a single matrix, based on their group membership (eg: SC, FC)
    
    Parameters:
    data_dict: dict. Contains [nsubj x nfeat] np.ndarray for each connectivity type (flavor)
    group_dict: dict. Contains a group for each connectivity type (flavor)
    group_to_combine: str. Group to combine (eg: 'SC', 'FC', 'SCFC'). Default='all'
    combine_type: str. 'mean' or 'concat'. If 'mean', will average across flavors. If 'concat', will concatenate across flavors. Default='mean'
    normalize: bool. If True, L2-normalize the combined data after averaging. Default=False
    
    Returns:
    data_combined: np.ndarray. [nsubj x nfeat] matrix of combined data
    """
    if group_to_combine=='SCFC':
        grouplist=['SC','FC']
    else:
        grouplist=[group_to_combine]
    
    if combine_type=='mean':
        data_combined=np.mean(np.stack([data_dict[c] for c in data_dict if group_dict[c] in grouplist],axis=-1),axis=-1)
    elif combine_type in ['concat','concatenate']:
        data_combined=np.concatenate([data_dict[c] for c in data_dict if group_dict[c] in grouplist],axis=1)
    elif combine_type.startswith('concat+pc'):
        concat_pc_num=int(combine_type.replace("concat+pc",""))
        data_temp=np.concatenate([data_dict[c] for c in data_dict if group_dict[c] in grouplist],axis=1)
        data_combined=PCA(n_components=concat_pc_num,random_state=0).fit_transform(data_temp)
    else:
        raise Exception("Unknown combine_type: %s" % (combine_type))
    
    if normalize:
        normfun=lambda x: x/np.sqrt(np.sum(x**2,axis=1,keepdims=True))
        data_combined=normfun(data_combined)
        
    return data_combined

############################# load krakencoder network ######################

if model_kraken  == 'sex_Krakencoder':
    checkpoint_file = '/LS3/krakenLS3dynICV403demofineSEX100_chkpt_SCFC2all_16flav_1419subj_pc256_240paths_latent128_latentunit_drop0.5_500ep_correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000_20251107_144040_ep000500.pt'
elif model_kraken == 'sex_site_Krakencoder':
    checkpoint_file = '/LS3/krakenLS3dynICV403demofineAGE0SEXSITEcond100_chkpt_SCFC2all_18flav_1419subj_zfeat_270paths_latent128_latentunit_drop0.5_500ep_correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000_20251107_010029_ep000500.pt'
ioxfm_file_list=['/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_SCifod2act_fs86_volnormicv_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_SCsdstream_fs86_volnormicv_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCcorr_fs86_hpf_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCcorr_fs86_hpfgsr_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCpcorr_fs86_hpf_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_SCifod2act_shen268_volnormicv_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_SCsdstream_shen268_volnormicv_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCcorr_shen268_hpf_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCcorr_shen268_hpfgsr_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCpcorr_shen268_hpf_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_SCifod2act_coco439_volnormicv_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_SCsdstream_coco439_volnormicv_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCcorr_coco439_hpf_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCcorr_coco439_hpfgsr_pc256_403train.npy',
                 '/LS3/ioxfm_dynseed/kraken_ls3_ioxfm_FCpcorr_coco439_hpf_pc256_403train.npy',
                 '/LS3/kraken_ls3_ioxfm_age_sex_site_zfeat_403train.npy']

inner_net, checkpoint_info = Krakencoder.load_checkpoint(checkpoint_file, eval_mode=True)
transformer_list, transformer_info_list = load_transformers_from_file(ioxfm_file_list)


#create new model that wraps the inner kraken model and includes PCA transforms from raw data
net=KrakenAdapter(inner_model=inner_net,
                data_transformer_list=[transformer_list[conntype] for conntype in checkpoint_info['input_name_list']],
                linear_polynomial_order=0, eval_mode=True)

conntypes = checkpoint_info['input_name_list'][:-3]

############################### load_data ################################
###### demograph ######
demograph = pd.read_csv("/midtier/cocolab/scratch/kwj2001/LS3/ls3_subjinfo_agesexcog_details_1419subj.csv")
kr_age = demograph['age']
kr_sex = demograph['sex'].map({'M': 1, 'F': 0})
df_demograph = pd.DataFrame({
    'age': kr_age,
    'sex_index': kr_sex})
df_demograph['age_category'] = pd.cut(df_demograph['age'], bins=bins, right=False)

##### raw 1419 data #####
FC_list = ['FCcorr_coco439_hpf', 'FCcorr_coco439_hpfgsr', 'FCcorr_fs86_hpf', 'FCcorr_fs86_hpfgsr', 'FCcorr_shen268_hpf', 'FCcorr_shen268_hpfgsr', 
            'FCpcorr_coco439_hpf', 'FCpcorr_fs86_hpf', 'FCpcorr_shen268_hpf']
SC_list = ['SCifod2act_coco439_volnormicv', 'SCifod2act_fs86_volnormicv', 'SCifod2act_shen268_volnormicv', 
            'SCsdstream_coco439_volnormicv', 'SCsdstream_fs86_volnormicv', 'SCsdstream_shen268_volnormicv']
c_list = FC_list + SC_list

if split0328 == True: ## (newYA: use fine-tuned krakencoder / compare the performance with fine-tuned 0328)
    Msplit = scipy.io.loadmat("/LS3/ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat",simplify_cells=True)


datadir = '/LS3/rawdata_dynseed'
Cflats = {}
Cflats_kr = {}

for conntype in c_list:
    inputfile="%s/ls3_%s_1419subj.mat" % (datadir,conntype) 
    Cdata=scipy.io.loadmat(inputfile,simplify_cells=True)['C'] #read the flavor data 
    nroi=Cdata[0].shape[0]
    trimask=np.triu_indices(nroi,1) 
    Cflat=np.vstack([x[trimask] for x in Cdata]) # extract the upper triangle (shape: n_subjects x n_edges)

    if split0328: # split like krakencoder0328
        Cflat_select = Cflat[Msplit['subjidx_test'],:]
        df_demo_select = df_demograph.iloc[Msplit['subjidx_test'],:].reset_index(drop=True) # select the fine-tuned 0328 held out data to use
        data_use_df = pd.concat([df_demo_select, pd.DataFrame(Cflat_select)], axis=1)
        
        if add_trainingYA == 'newYA_train':    
            Cflats_trkr = Cflat[Msplit['subjidx_train'],:]
            df_demo_trkr = df_demograph.iloc[Msplit['subjidx_train'],:].reset_index(drop=True)
            Cflats_valkr = Cflat[Msplit['subjidx_val'],:]
            df_demo_valkr = df_demograph.iloc[Msplit['subjidx_val'],:].reset_index(drop=True)
            Cflats_df_trkr = pd.concat([df_demo_trkr, pd.DataFrame(Cflats_trkr)], axis=1) 
            Cflats_df_valkr = pd.concat([df_demo_valkr, pd.DataFrame(Cflats_valkr)], axis=1)
            Cflats_df_trkr = pd.concat([Cflats_df_trkr, Cflats_df_valkr])
            grouped_tr_kr = Cflats_df_trkr.groupby('age_category', observed=False)
            cage_kr = {age_category: group for age_category, group in grouped_tr_kr}
            Cflats_kr[conntype] = {'data': cage_kr}
    elif not split0328:
        data_use_df = pd.concat([df_demograph, pd.DataFrame(Cflat)], axis=1)  
    grouped = data_use_df.groupby('age_category', observed=False)        
    cage = {age_category: group for age_category, group in grouped}
    Cflats[conntype] =  {'data': cage,
        'trimask': trimask}

############################## mask data ####################################
#### load yeo atlas
Myeo=scipy.io.loadmat('/home/HCP_data/fc_merged_atlasconcat_justyeo.mat',simplify_cells=True)
whichyeo='yeo7'
yeonum=7+2
yeonames=Myeo['%s_names' % (whichyeo)]
conntype_roi_yeoidx={}
for conntype in conntypes:
    if 'fs86' in conntype:
        conntype_roi_yeoidx[conntype]=Myeo['%s_index' % (whichyeo)]['fs86']
    elif 'shen268' in conntype:
        conntype_roi_yeoidx[conntype]=Myeo['%s_index' % (whichyeo)]['shen268']
    elif 'cocommpsuit439' in conntype or 'coco439' in conntype:
        conntype_roi_yeoidx[conntype]=Myeo['%s_index' % (whichyeo)]['cocommpsuit439']
    
## a list of all the yeo pairs
yeopair_name_list=[]
yeopair_index_list=[]
yeopair_name_list.append('INCLUDE:ALL')
yeopair_index_list.append((-1,-1))
    
#### mask ALL connections to/from each yeo network #####
for iy in range(1,yeonum+1):
    if do_exclude_mask:
        yeopair_index_list.append((iy,-1))
        yeopair_name_list.append('EXCLUDE:%s' % (yeonames[iy-1]))
    if do_include_mask:
        yeopair_index_list.append((iy,-1))
        yeopair_name_list.append('INCLUDE:%s' % (yeonames[iy-1]))

#### select each network to mask
conntype_edge_yeomask={}
numroi_alltypes=[]
for conntype in conntypes:
    trimask=Cflats[conntype]['trimask']
    yeo1=conntype_roi_yeoidx[conntype][trimask[0]]
    yeo2=conntype_roi_yeoidx[conntype][trimask[1]]
    yeopairmask=np.zeros((len(yeopair_name_list),len(yeo1)))
        
    for ypidx, (iy,jy) in enumerate(yeopair_index_list):
        if (iy,jy)==(-1,-1):
            yeopairmask[ypidx,:]=1
        elif jy==-1:
            m=(yeo1==iy) | (yeo2==iy) # if belong to the network iy, mark as 1
            yeopairmask[ypidx,m]=1
    conntype_edge_yeomask[conntype]=yeopairmask

################################ split between ages ##################################
masked_results = {} 
for i in range(len(bins)-1): # each age range
    np.random.seed(seed + i)
    interval = pd.Interval(left=bins[i], right=bins[i+1], closed='left')

    result_ypdix = {} # results for each include or exclude network
    for ypidx in range(len(yeopair_index_list)): # include or exclude each network
        test_acc = np.empty((outer_folds, 3)) # fusion; fusionFC, fusionSC
        test_pred_enc = []
        test_pred_encFC = []
        test_pred_encSC = []

        result_fusionFCSC = {}
        for outer_iter in range(outer_folds):
            rng = seed + outer_iter
            np.random.seed(rng)

            # only consider the situation when split0328 == True: 
            # use new YA (low dimensional data predicted by krakencoder0328 or original krakencoder)
            encoded_alltypes={}
            for encidx, conntype in enumerate(conntypes):
                grp = Cflats[conntype]['data'][interval]
                males = grp[grp['sex_index'] == 1]
                females = grp[grp['sex_index'] == 0]

                m_tr, m_te = train_test_split(males, test_size = int(testsize/2), random_state = rng) 
                f_tr, f_te = train_test_split(females, test_size = int(testsize/2), random_state = rng)

                if add_trainingYA == 'newYA_train': 
                    kr_grp = Cflats_kr[conntype]['data'][interval]                    
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

                combined_df = pd.concat([train_df, test_df], ignore_index=True)
                X = combined_df.iloc[:, 3:]
                y = combined_df['sex_index']

                n_train = len(train_df)
                n_test  = len(test_df)
                is_train = np.hstack([np.ones(len(train_df), dtype=bool),
                                      np.zeros(len(test_df),  dtype=bool)])

                if is_meanfit_shift:
                    adapted = generate_adapt_transformer(input_data=X.values, target_data=transformer_info_list[conntype], 
                                                            input_data_fitsubjmask=is_train, adapt_mode='meanfit+meanshift')
                    data_masked =  adapted.transform(X.values).numpy()
                else:
                    data_masked = X.values.copy()

                if yeopair_name_list[ypidx].startswith('INCLUDE:'):
                    edgemask_replace=conntype_edge_yeomask[conntype][ypidx,:]==0 # replace those not belong to the network
                elif yeopair_name_list[ypidx].startswith('EXCLUDE:'):
                    edgemask_replace=conntype_edge_yeomask[conntype][ypidx,:]==1 # replace those belong to the network

                if replace_with=='mean':
                    trainmean=np.mean(data_masked[is_train,:],axis=0,keepdims=True)
                    data_masked[:,edgemask_replace]=trainmean[:,edgemask_replace]

                with torch.no_grad():
                    encoded_alltypes[conntype]=net(data_masked,encoder_index=encidx, decoder_index=-1)

            conngroups=get_conngroup_dict(conntypes)
            enc_cat=None
            enc_mean=combine_input_groups(encoded_alltypes,conngroups, group_to_combine='SCFC', combine_type='mean', normalize=False)
            enc_FCmean=combine_input_groups(encoded_alltypes,conngroups, group_to_combine='FC', combine_type='mean', normalize=False)
            enc_SCmean=combine_input_groups(encoded_alltypes,conngroups, group_to_combine='SC', combine_type='mean', normalize=False)
        
            Data_masked={}
            Data_masked['enc_mean']=enc_mean
            Data_masked['enc_FCmean']=enc_FCmean
            Data_masked['enc_SCmean']=enc_SCmean

            for fushion_type in ['enc_mean', 'enc_FCmean', 'enc_SCmean']:
                data_type = Data_masked[fushion_type]
                X_tr, X_te = data_type[is_train,:], data_type[~is_train,:]
                y_train, y_test = y[is_train], y[~is_train]
                      
                inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=rng)
                model = LogisticRegression(penalty='l2', max_iter=1000, random_state=rng)
                grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_tr, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_te)

                if fushion_type == 'enc_mean':
                    test_acc[outer_iter, 0] = accuracy_score(y_test, y_pred)
                    test_pred_enc.append([y_test, y_pred])
                elif fushion_type =='enc_FCmean':
                    test_acc[outer_iter, 1] = accuracy_score(y_test, y_pred)
                    test_pred_encFC.append([y_test, y_pred])
                else:
                    test_acc[outer_iter, 2] = accuracy_score(y_test, y_pred)
                    test_pred_encSC.append([y_test, y_pred])

        result_ypdix[yeopair_name_list[ypidx]] = {
            'test_acc_fusion': test_acc[:,0],
            'test_acc_FC': test_acc[:,1],
            'test_acc_SC': test_acc[:,2],
            'test_result_fusion': test_pred_enc,
            'test_result_FC': test_pred_encFC,
            'test_result_SC': test_pred_encSC
        }
    masked_results[str(bins[i])+'-'+str(bins[i+1])] = result_ypdix

if is_meanfit_shift:
    with open('/home/out_log/results_HCPdata_Kraken/newYA95/sensitivity/results1110/sensitivity_' + str(model_kraken) + '_' + str(bins) + '_stratifiedCV_split0328_test'+ str(testsize) + '_add' + str(add_trainingYA)+ '_downsampledtrain'+ str(train_downsample_size*2) + '_nwlevel_meanshift.pkl', 'wb') as pickle_file:
        pickle.dump(masked_results, pickle_file)    
else:
    with open('/home/out_log/results_HCPdata_Kraken/newYA95/sensitivity/results1110/sensitivity_' + str(model_kraken) + '_' + str(bins) + '_stratifiedCV_split0328_test'+ str(testsize) + '_add' + str(add_trainingYA)+ '_downsampledtrain'+ str(train_downsample_size*2) + '_nwlevel.pkl', 'wb') as pickle_file:
        pickle.dump(masked_results, pickle_file)    
