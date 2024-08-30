'''
    Author: Daniel Rawlinson
    Email: daniel.rawlinson@unimelb.edu.au
    Affiliation: Peter Doherty Institute for Infection and Immunity
    
    '''
###### imports ######

import os
import sys
import scanpy as sp
import pickle
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import typing
from scipy.stats import chi2

###### constants ######
#None

###### functions ######

def infill_missing_features(scanpy_dat_RNA, model: typing.Union[Lasso, LassoCV, list]):
    '''
    Make anndata input compatible with trained model by infilling missing features with zero (0).
    Output is anndata but will lose all .var data associated with input.
        '''
    if isinstance(model, (Lasso, LassoCV)):
        model_features = set(model.feature_names_in_)
    elif isinstance(model, list):
        model_features = set(model)
    else:
        raise TypeError('Model is not Lasso, LassoCV or list of features. Other model types not implemented yet')
   
    missing_feats = model_features - set(scanpy_dat_RNA.var_names)
    sys.stderr.write('{} features missing from test dataset. In-filling with zeros'.format(len(missing_feats)))
    scanpy_dat_w_infill = scanpy_dat_RNA.to_df().join(pd.DataFrame(
        {c:np.float32(0) for c in missing_feats
        }, index=scanpy_dat_RNA.to_df().index))
    scanpy_dat_w_infill = scanpy_dat_w_infill.reindex(columns = model_features)   
    return(sp.AnnData(scanpy_dat_w_infill, obs = scanpy_dat_RNA.obs))

def split_AnnData(scanpy_dat_RNA, scanpy_dat_ADT, test_portion = 0.2, highly_variable_only = False, renormalise = False, ax = 0):
    '''
    Split ADT and RNA anndata into test and train partitions. Returns a dictionary of {train: [X, y], test: [X, y]}
    Set highly_variable to reduce RNA feature set to those listed as highly variable in the anndata metadata.
    Set renormalise to perform normalisation on the split partitions.
        '''
    check_order(scanpy_dat_RNA, scanpy_dat_ADT)
    
    if highly_variable_only:
        scanpy_dat_RNA = subset_to_highly_variable(scanpy_dat_RNA) #won't work in this location because it removed features that are needed for normalisation
    train_names, test_names = model_selection.train_test_split(scanpy_dat_RNA.obs_names, test_size = test_portion, random_state=42)
    X_train, X_test = [scanpy_dat_RNA[train_names,:], scanpy_dat_RNA[test_names,:]]
    y_train, y_test = [scanpy_dat_ADT[train_names,:], scanpy_dat_ADT[test_names,:]]
    out = {'train': [X_train, y_train], 'test': [X_test, y_test]}
    
    if renormalise:
        out = {k: renormalise_subsets(v, ax = ax) for k, v in out.items()}
    return(out)

def select_by_metadata(scanpy_dat_RNA, scanpy_dat_ADT, column_name, column_value, highly_variable_only = False, renormalise = False):
    '''
    Split ADT and RNA anndata into test and train by a named column in the metadata. 
    Returns a dictionary of {train: [X, y], test: [X, y]} where the test data is constituted of samples with named metadata: scanpy_dat_RNA.var[column_name] == column_value.
    Set column_value . Set renormalise to perform normalisation on the split partitions.
        '''
    check_order(scanpy_dat_RNA, scanpy_dat_ADT)
    
    #enforce column is a categorical one
    if not pd.api.types.is_numeric_dtype(scanpy_dat_RNA[column_name]):
        raise Exception('Chosen column dtype is numeric. Check you are choosing the desired variable and convert dtype if necessary' )
    
    if highly_variable_only:
        scanpy_dat_RNA = subset_to_highly_variable(scanpy_dat_RNA)
    
    X_test = scanpy_dat_RNA[scanpy_dat_RNA.var[column_name] == column_value,:]
    y_test = scanpy_dat_ADT[X_train.obs_names,:]

    X_train = scanpy_dat_RNA[-X_test.obs_names,:]
    y_train = scanpy_dat_ADT[-y_test.obs_names,:]
    
    out = {'train': [X_train, y_train], 'test': [X_test, y_test]}
    return(out)   


def renormalise_subsets(scanpy_dat_split, ax = 0): #RNA index 0, ADT index 1
    if not all([isinstance(sc, sp.AnnData) for sc in scanpy_dat_split]):
        raise TypeError('Input is not iterable of AnnData objects'.format())
    check_order(scanpy_dat_split[0], scanpy_dat_split[1])
     
    X_adata = scanpy_dat_split[0].copy()
    X_adata.X = X_adata.raw.X #restore raw to active layer
    sp.pp.normalize_total(X_adata, target_sum = 10000, inplace = True)
    
    y_adata = scanpy_dat_split[1].copy()
    y_adata.X = y_adata.raw.X
    y_adata = normalize_geometric(y_adata, ax = ax, target_sum=10000)
    return([X_adata, y_adata]) 

def id_mouse_genes(scanpy_dat_RNA, var_prefix = 'MOUSE_'):
    scanpy_dat_RNA.var['mouse'] = scanpy_dat_RNA.var_names.str.startswith(var_prefix)
    return(scanpy_dat_RNA)

def id_mouse_cells(scanpy_dat_RNA):
    sp.pp.calculate_qc_metrics(scanpy_dat_RNA, expr_type='counts', qc_vars='mouse', inplace=True, log1p=False,percent_top=None)
    return(scanpy_dat_RNA)

def remove_mouse_features(scanpy_dat_RNA):
    scanpy_dat_RNA = scanpy_dat_RNA[:,-scanpy_dat_RNA.var['mouse']]
    return(scanpy_dat_RNA)

def remove_mouse_cells(scanpy_dat_RNA, plot = True):
    scanpy_dat_RNA = scanpy_dat_RNA[scanpy_dat_RNA.obs.pct_counts_mouse < 90, :]
    if plot == True:
        sns.violinplot(scanpy_dat_RNA.obs['pct_counts_mouse'])
    return(scanpy_dat_RNA)
    
def match_cell_barcodes(scanpy_dat_RNA, scanpy_dat_ADT):
    common_cells = list(set(scanpy_dat_ADT.obs_names) & set(scanpy_dat_RNA.obs_names))
    scanpy_dat_RNA = scanpy_dat_RNA[common_cells]
    scanpy_dat_ADT = scanpy_dat_ADT[common_cells]
    return([scanpy_dat_RNA, scanpy_dat_ADT])

def check_common_barcodes(scanpy_dat_RNA, scanpy_dat_ADT):
    return(scanpy_dat_ADT.obs_names == scanpy_dat_RNA.obs_names)

def subset_to_highly_variable(adata_RNA):
    adata_RNA = adata_RNA[:,adata_RNA.var['highly_variable']]
    return(adata_RNA)

def _clr(x, target_sum = 10000): #added in target_sum so every normalised dataset is on the same scale
    x=target_sum*(x/sum(x))
    log_sum = np.sum(np.log1p(x[x>0]))
    clr_denom = np.exp(log_sum / len(x)) #divide by len(x) because x=0 ought to pull mean down.
    return(np.log1p(x/ clr_denom))


def normalize_geometric(adata, ax = 0, target_sum = 10000): #default ax=0 is to normalise within each cell
    inp_arr = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X 
    adata.X = np.apply_along_axis(func1d = _clr 
                                  , axis = ax, arr = inp_arr, target_sum = target_sum)
    return(adata)


def ADT_ridge_plots(scanpy_dat_ADT, selected_ADTs = None):
    '''If this fails with error that index is out of bounds for ndim 0, ensure that scanpy_dat_ADT.X is in np.array format'''
    
    pd_ADT = scanpy_dat_ADT.to_df()
    
    if selected_ADTs:
        pd_ADT = pd_ADT[selected_ADTs]
    
    pd_ADT = pd_ADT.melt(var_name='ADT', value_name= 'clr_count')
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(pd_ADT, row="ADT", hue = 'ADT', aspect = 9, height = 1.2)
    g.map_dataframe(sns.kdeplot, x="clr_count")
    g.set_titles("")
    g.set(yticks=[])
    g.despine(left=True)
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g.map_dataframe(sns.kdeplot, x="clr_count", fill=True, alpha=1)
    g.map_dataframe(sns.kdeplot, x = 'clr_count', color = 'black')

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, color='black', fontsize=13,
                ha="left", va="center", transform=ax.transAxes)
        ax.set(ylabel = '')

    # def add_rain(x, color, label):
    #     curr_ax = plt.gca()
    #     sns.stripplot(x = x, ax= curr_ax)

    g.map(label, "ADT")

    #g.map(add_rain, 'clr_count') #can just change this to map sns.stripplot. But that still flips the kdeplot over.
    #gave up on this


    g.fig.subplots_adjust(hspace = -.5)

    g.set_titles("")
    g.set(yticks=[])
    g.despine(left=True)
    
    return(g)

def plot_testVtrain_lasso(ADT_models, RNA_test, ADT_test, plot = True):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    score_dict = get_scores(ADT_models, RNA_test, ADT_test)
    for adt in ADT_models.keys():
        corr_df = pd.DataFrame()
        corr_df['y_preds'] = get_preds(ADT_models[adt], RNA_test)
        corr_df['y_true'] = ADT_test[:,adt].X
        if plot == True:
            fig, axs = plt.subplots(nrows = 1, ncols= 3, figsize = (12,5), sharey = False)
            fig.suptitle(adt, x = -0.05, y = 0.5, ha = 'left', va = 'bottom')
            sort_true = corr_df['y_true'].sort_values()
            axs[0].scatter(corr_df['y_true'], corr_df['y_preds'])
            axs[0].set_title('Scatter')
            axs[1].scatter(range(len(sort_true)), corr_df['y_preds'][sort_true.index], color = 'orange')
            axs[1].set_title('Rank vs predicted')

            rankvscatter_loess = lowess(exog = range(len(sort_true)), endog = corr_df['y_preds'][sort_true.index], frac = 0.3)    
            axs[1].plot(rankvscatter_loess[:,0], rankvscatter_loess[:,1])

            sort_pred = corr_df['y_preds'].sort_values()
            rank_compare = pd.DataFrame({'true' : sort_true.index, 'pred' :  sort_pred.index}) #so that ith cell is before jth cell if prot(i) < prot(j) in each column

            rank_plot = pd.DataFrame(index=range(len(sort_pred)),columns=(0,1))
            for i in range(len(sort_pred)):
                true_rank = rank_compare['true'].tolist().index(i)
                pred_rank = rank_compare['pred'].tolist().index(i)
                rank_plot.iloc[i] = [true_rank, pred_rank]

            axs[2].scatter(rank_plot[0], rank_plot[1], color = 'red')
            axs[2].set_title('Rank vs rank')

            rankvrank_loess = lowess(exog = rank_plot[0], endog = rank_plot[1], frac = 0.3)    
            axs[2].plot(rankvrank_loess[:,0], rankvrank_loess[:,1])
    return(score_dict)

def check_order(adata_RNA, adata_ADT):
    assert adata_RNA.shape[1] > adata_ADT.shape[1], 'ADT object has more features than RNA object. Check the order of inputs has not been jumbled'
    return()

def scores_heatmap(scores_output, reorder = False):
    scores_df = pd.DataFrame(scores_output)
    scores_df = scores_df.set_index('adt')
    #if reorder:
    #    scores_df = scores_df.reindex(
    
    sns.set(font_scale = 0.7 )
    plt.figure(figsize = (4,3))
    ax = sns.heatmap(scores_df, annot=True, cmap=sns.light_palette("seagreen", as_cmap=True), xticklabels=['Pearson', 'Spearman', '$R^2$'], cbar = False)
    ax.set(ylabel = 'ADT')
    plt.tight_layout()
    return()

def eval_residuals(ADT_models, adt, RNA_test, ADT_test):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,5))
    fig.suptitle(adt)
    fig.tight_layout()
    preds = get_preds(ADT_models[adt], RNA_test)
    true = ADT_test[:,adt].X.ravel()
    residuals = true - preds
    ##QQ
    sm.qqplot(residuals,  line = '45', ax = ax[0])
    #heteroescadidicity
    sns.residplot(x = true, y = preds, lowess = True, line_kws = {'color':'red'}, ax = ax[1])
    ax[1].set_xlabel('Fitted value')
    ax[1].set_ylabel('Residuals')
    #Residual density
    sns.kdeplot(x = residuals, ax = ax[2])
    ax[2].set_xlabel('Residuals')
    #
    
    # 3 plots here. QQ-plot, heteroescadicity, resdiual histogram
    
    
def get_preds(ADT_model, RNA_test):
    preds = ADT_model.predict(X = RNA_test[:,ADT_model.feature_names_in_].to_df())
    return(preds)
    
def get_scores(ADT_models, RNA_test, ADT_test): #This and the plotting function and the eval residuals could be better factored. There is double-up, including the calling of predictions from the model
    score_dict = {'adt': [], 'pearson_score' : [], 'spearman_score' :[], 'r2_score' : [] }
    for adt in ADT_models.keys():
        score_dict['adt'].append(adt)
        corr_df = pd.DataFrame()
        corr_df['y_preds'] = get_preds(ADT_models[adt], RNA_test)
        corr_df['y_true'] = ADT_test[:,adt].X
        score_dict['pearson_score'].append(corr_df.corr(method = 'pearson').iloc[0,1])
        score_dict['spearman_score'].append(corr_df.corr(method = 'spearman').iloc[0,1])
        score_dict['r2_score'].append(r2_score(corr_df['y_true'], corr_df['y_preds']))
    return(score_dict)

def ADT_gaussian_mixture_BIC(scanpy_dat_ADT, adt_name, score_method = 'gmm_bic_score'):
    adt_values = scanpy_dat_ADT[:,adt_name].X.ravel().reshape(-1, 1)
    param_grid = {
    "n_components": range(1, 5),
    "covariance_type": ["full"]#['full'] #just going to assume 'full'
    }
    
    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)
        
    gs = model_selection.GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score)
    gs.fit(adt_values)
    
    df = pd.DataFrame(gs.cv_results_)[["param_n_components", "param_covariance_type", "mean_test_score"]]
    df["mean_test_score"] = -df["mean_test_score"] #re-inversed here so that we actually aim to minimise the BIC when we inspect it visually
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": 'BIC score',
        }
    )
    df.sort_values(by="BIC score").head()
    print(df)
    return(gs)

def ADT_gaussian_mixture_fitrange(scanpy_dat_adt, adt_name):
    '''This is to fit GMM with a varying number of components to the ADT data. To later compare models with Likelihood Ratio Test
    Remember this only works for *nested* models'''
    n_components = range(1,5)
    adt_values = scanpy_dat_adt[:,adt_name].X.ravel().reshape(-1, 1)
    adt_train, adt_test = model_selection.train_test_split(adt_values, random_state=42, test_size = 0.2)
    gmms = {}
    for n_comp in n_components:
        gmms[n_comp] = GaussianMixture(n_components = n_comp, covariance_type = 'full').fit(adt_train)

    def likelihood_ratio(llmin, llmax):
        return(2*(llmax-llmin)) #equivalent to -2*log(likelihood_nought/likelihood_test)
    
    LRT_evals = pd.DataFrame(columns = ['comparison','LR','p'])
    for n_comp, model in gmms.items():
        if n_comp ==1:
            loglh = None #init log likelihood
        loglh_nested = loglh #store previous log-likelihood
        loglh = model.score(adt_test)
        gs_df = model.get_params()
        if n_comp != 1:
            LR = likelihood_ratio(loglh_nested, loglh)
            prob = chi2.sf(LR, 2) #deg of freedom is always 2 as n_components is jumping by 1. So there is always an extra SD and mean. degrees of freedom equal to the difference in dimensionality of Θ and Θ nought 
            LRT_evals.loc[len(LRT_evals.index)] = ['{}vs{}'.format(n_comp-1, n_comp), LR, prob ]
    print(LRT_evals)
    return(gmms)

def ADT_gaussian_mixture_plot(scanpy_dat_ADT, adt_name, gmm):
    X = scanpy_dat_ADT[:,adt_name].X.ravel().reshape(-1,1)
    color_iter = sns.color_palette("tab10", 4)[::-1]
    Y_ = gmm.predict(X)
    total_bins = 100
    for i, (mean, cov, col) in enumerate(zip(gmm.means_,
                                                gmm.covariances_,
                                                color_iter)):
        subset = X[Y_ == i, 0].copy()
        tot_width = X.max() - X.min()
        subset_width = subset.max() - subset.min()
        n_bins = int(100 * subset_width / tot_width)
        plt.hist(subset, n_bins, color=col, alpha = 0.5, stacked = True)
    plt.title(adt_name)
    plt.show()

def denoise_ADT_with_dsb(scanpy_dat_ADT):
    return(scanpy_dat_ADT)