"""
All functions except covbat are forked from
https://github.com/brentp/combat.py
combat function modified to enable correction without empirical Bayes
covbat function written by Andrew Chen (andrewac@pennmedicine.upenn.edu)
"""
import pandas as pd
import patsy
import sys
import numpy.linalg as la
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def adjust_nums(numerical_covariates, drop_idxs):
    # if we dropped some values, have to adjust those with a larger index.
    if numerical_covariates is None: return drop_idxs
    return [nc - sum(nc < di for di in drop_idxs) for nc in numerical_covariates]

def design_mat(mod, numerical_covariates, batch_levels):
    # require levels to make sure they are in the same order as we use in the
    # rest of the script.
    design = patsy.dmatrix("~ 0 + C(batch, levels=%s)" % str(batch_levels),
                                                  mod, return_type="dataframe")

    mod = mod.drop(["batch"], axis=1)
    numerical_covariates = list(numerical_covariates)
    sys.stderr.write("found %i batches\n" % design.shape[1])
    other_cols = [c for i, c in enumerate(mod.columns)
                  if not i in numerical_covariates]
    factor_matrix = mod[other_cols]
    design = pd.concat((design, factor_matrix), axis=1)
    if numerical_covariates is not None:
        sys.stderr.write("found %i numerical covariates...\n"
                            % len(numerical_covariates))
        for i, nC in enumerate(numerical_covariates):
            cname = mod.columns[nC]
            sys.stderr.write("\t{0}\n".format(cname))
            design[cname] = mod[mod.columns[nC]]
    sys.stderr.write("found %i categorical variables:" % len(other_cols))
    sys.stderr.write("\t" + ", ".join(other_cols) + '\n')
    return design

"""Correction of *Cov*ariance *Bat* effects

Parameters
----------
data : pandas.DataFrame
    A (n_features, n_samples) dataframe of the expression or methylation
    data to batch correct
batch : pandas.Series
    A column corresponding to the batches in the data, with index same as
    the columns that appear in ``data``
model : patsy.design_info.DesignMatrix, optional
    A model matrix describing metadata on the samples which could be
    causing batch effects. If not provided, then will attempt to coarsely
    correct just from the information provided in ``batch``
numerical_covariates : list-like
    List of covariates in the model which are numerical, rather than
    categorical
pct_var : numeric
    Numeric between 0 and 1 indicating the percent of variation that is
    explained by the adjusted PCs
n_pc : numeric
    If >0, then this specifies the number of PCs to adjust. Overrides pct_var

Returns
-------
corrected : pandas.DataFrame
    A (n_features, n_samples) dataframe of the batch-corrected data
"""
def covbat(data, batch, model=None, numerical_covariates=None, pct_var=0.95, n_pc=0):
############### getDataDict (Getting design) ###############
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).iteritems() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
        for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

########################################################
#################### getStandardizedData ###############
    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((int(n_array), 1)) / int(n_array))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T    ### stand_mean here include mod.mean (mod.mean is np.dot(tmp, B_hat).T)

    s_data = ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))
#######################################################
###################   L/S estimates  ###################
    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        #batches = [list(model.columns).index(b) for b in batches]
        delta_hat.append(s_data[batch_idxs].var(axis=1))

###############################################################
############## eb estimators (get final estimators) ###########
    gamma_bar = gamma_hat.mean(axis=1) 
    t2 = gamma_hat.var(axis=1)
   
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        #print '18 20 22 28 29 31 32 33 35 40 46'
        #print batch_info[batch_id]

        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                     delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

##############################################################
##################### getCorrectedData  ######################
## Note: standard mean is not added back to bayesdata yet, will be added back at the end of covbat function
    for j, batch_idxs in enumerate(batch_info):

        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom =  np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

        bayesdata[batch_idxs] = numer / denom
   
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    # not adding back stand_mean yet
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array))))

############### get estimates from (getCorrectedData) ###############
    # estimates: delta_hat, gamma_star, delta_star, gamma_bar, t2, a_prior, b_prior, stand_mean, var_pooled
    estimates = {
        'gamma_hat': gamma_hat, ## L/S estimates
        'delta_hat': delta_hat, ## L/S estimates
        'gamma_star': gamma_star, ## EB estimates (final estimators)
        'delta_star': delta_star, ## EB estimates
        'gamma_bar': gamma_bar, ## EB estimates
        't2': t2, ## EB estimates
        'a_prior': a_prior, ## EB estimates
        'b_prior': b_prior, ## EB estimates
        'stand_mean': stand_mean, ## standard mean (w mod.mean)
        'grand_mean': grand_mean, ## grand mean
        'var_pooled': var_pooled, ## standard variance
        'beta_hat': B_hat, ## b_hat
        } # compared with yeo script; missing batch, eb (T/F), parametric (T/F), mean.only (T/F) etc estimates 

############### covbat ################    
    ### note: eb=False in the combat function below to avoid empirical Bayes adjustment in CovBat step; only eb=True has parametric 
    # CovBat step: PCA then ComBat without EB on the scores
    # comdata = data.T
    comdata = bayesdata.T
    bmu = np.mean(comdata, axis=0)
    # standardize data before PCA
    scaler = StandardScaler()
    comdata = scaler.fit_transform(comdata)
    
    pca = PCA()
    pca.fit(comdata)
    pc_comp = pca.components_
    full_scores = pd.DataFrame(pca.fit_transform(comdata)).T
    full_scores.columns = data.columns

    var_exp=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
    npc = np.min(np.where(var_exp>pct_var))+1
    if n_pc > 0:
        npc = n_pc
    scores = full_scores.loc[range(0,npc),:]
    combat_result = combat(scores, batch, model=None, eb = True) # default is False in original codes
    scores_com = combat_result['dat_combat']
    combat_estimates = combat_result['combat_estimates']
    full_scores.loc[range(0,npc),:] = scores_com

    x_covbat = bayesdata - bayesdata # create pandas DataFrame to store output
    proj = np.dot(full_scores.T, pc_comp).T
    x_covbat += scaler.inverse_transform(proj.T).T
    x_covbat += stand_mean #(stand_mean includes mod.mean, which is np.dot(tmp, B_hat).T)
 
    results = {
        'dat_covbat': x_covbat,
        'dat_pca_est': pca,
        'dat_covcomb_est': combat_estimates,
        'dat_npc': npc,
        'dat_combat': comdata, # Note: standard mean (include mode mean) is not added back
        'dat_estimates': estimates,
        'scaler': scaler,
        'batch_levels': batch_levels
    }

    return results


def covbat_test(dat, test_batch, estimates, pca_est, npc, covcomb_est, train_batch_levels, scaler, numerical_covariates, model=None):
    train_batch_levels = train_batch_levels
    # Check if all sites in the new data were in the training data
    test_batch_levels = sorted(test_batch.unique())
    for batch in test_batch_levels:
        if batch not in train_batch_levels:
            raise ValueError(f"Batch {batch} not found in training data")        
    n_batch = len(test_batch_levels)
    #### note: for structural connectome, need to check missing values

    # Step 0: standardize data
    var_pooled = estimates['var_pooled']
    #stand_mean = estimates['stand_mean']
    gamma_star = estimates['gamma_star']
    delta_star = estimates['delta_star']
    B_hat = estimates['beta_hat']
    grand_mean = estimates['grand_mean']
    pca = pca_est

    # batch design matrix for test data
    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(test_batch)
    else:
        model = pd.DataFrame({'batch': test_batch})
##########
    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).iteritems() if inter == True]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
        for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)
    batch_design = design[design.columns[:n_batch]]
    ########### standardize test data ################
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T    ### stand_mean here include mod.mean (mod.mean is np.dot(tmp, B_hat).T)

    s_data = ((dat - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))
    bayesdata = s_data.copy()
    ########### get corrected data ###########
    for j, batch_idxs in enumerate(batch_info):
        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom =  np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)
        bayesdata[batch_idxs] = numer / denom
   
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    # not adding back stand_mean yet
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array))))

    # 1. perform PCA to get top npc PCs
    scaler = scaler
    bayesdata_std = scaler.transform(bayesdata.T)
    full_scores = pd.DataFrame(pca.transform(bayesdata_std)).T
    full_scores.columns = dat.columns
    pc_comp = pca.components_

    # 2. Harmonize top npc PCs using covcomb_est
    scores = full_scores.loc[range(0,npc),:]
    stand_mean_combat = covcomb_est['stand_mean']   
    var_pooled_combat = covcomb_est['var_pooled']
    delta_star_combat = covcomb_est['delta_star']
    gamma_star_combat = covcomb_est['gamma_star']

    # 3. combat estimates from training data are used to adjust the test data
    scores_combat = ((scores - stand_mean_combat[:, 0][:, None]) / np.dot(np.sqrt(var_pooled_combat), np.ones((1, int(n_array)))))
   
    for j, batch_idxs in enumerate(batch_info):
        dsq = np.sqrt(delta_star_combat[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom =  np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(scores_combat[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star_combat).T)
        scores_combat[batch_idxs] = numer / denom

    vpsq_combat = np.sqrt(var_pooled_combat).reshape((len(var_pooled_combat), 1))
    scores_combat = scores_combat * np.dot(vpsq_combat, np.ones((1, int(n_array)))) + stand_mean_combat[:,0][:,None]

    #########################################
    full_scores2 = full_scores.copy()
    full_scores2.loc[range(0,npc),:] = scores_combat

    x_covbat = bayesdata - bayesdata # create pandas DataFrame to store output
    proj = np.dot(full_scores2.T, pc_comp).T
    x_covbat += scaler.inverse_transform(proj.T).T
    x_covbat += stand_mean 

    return x_covbat

#######################################################

def combat(data, batch, model=None, numerical_covariates=None, eb=True):
    """Correct for batch effects in a dataset

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_features, n_samples) dataframe of the expression or methylation
        data to batch correct
    batch : pandas.Series
        A column corresponding to the batches in the data, with index same as
        the columns that appear in ``data``
    model : patsy.design_info.DesignMatrix, optional
        A model matrix describing metadata on the samples which could be
        causing batch effects. If not provided, then will attempt to coarsely
        correct just from the information provided in ``batch``
    numerical_covariates : list-like
        List of covariates in the model which are numerical, rather than
        categorical
    eb : logical
        Should empirical Bayes adjustments be made, if FALSE then gamma_hat
        and delta_hat are used as correction

    Returns
    -------
    corrected : pandas.DataFrame
        A (n_features, n_samples) dataframe of the batch-corrected data
    """
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).iteritems() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
            for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((int(n_array), 1)) / int(n_array))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))

    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        #batches = [list(model.columns).index(b) for b in batches]
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1) 
    t2 = gamma_hat.var(axis=1)
   

    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        #print '18 20 22 28 29 31 32 33 35 40 46'
        #print batch_info[batch_id]

        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                     delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)


    for j, batch_idxs in enumerate(batch_info):
        if eb:
            dsq = np.sqrt(delta_star[j,:])
            dsq = dsq.reshape((len(dsq), 1))
            denom =  np.dot(dsq, np.ones((1, n_batches[j])))
            numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

            bayesdata[batch_idxs] = numer / denom
        else:
            gamma_hat = np.array(gamma_hat)
            delta_hat = np.array(delta_hat)
            
            dsq = np.sqrt(delta_hat[j,:])
            dsq = dsq.reshape((len(dsq), 1))
            denom =  np.dot(dsq, np.ones((1, n_batches[j])))
            numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_hat).T)

            bayesdata[batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
 
    combat_estimates = {
        'gamma_hat': gamma_hat, ## L/S estimates
        'delta_hat': delta_hat, ## L/S estimates
        'gamma_star': gamma_star, ## EB estimates (final estimators)
        'delta_star': delta_star, ## EB estimates
        'gamma_bar': gamma_bar, ## EB estimates
        't2': t2, ## EB estimates
        'a_prior': a_prior, ## EB estimates
        'b_prior': b_prior, ## EB estimates
        'stand_mean': stand_mean, ## standard mean (w mod.mean)
        'var_pooled': var_pooled, ## standard variance
        'beta_hat': B_hat ## b_hat
    } # compared with yeo script; missing batch, mod, eb (T/F), parametric (T/F), mean.only (T/F) etc estimates 

    combat_results = {
        'dat_combat': bayesdata, 
        'combat_estimates': combat_estimates
    }
    return combat_results

def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        #print g_hat.shape, g_bar.shape, t2.shape
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.values.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)
       
        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

    

def aprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (2 * s2 +m**2) / s2

def bprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)
