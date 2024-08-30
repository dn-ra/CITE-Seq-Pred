#!/~/.conda/envs/scvi/bin/python
'''
Python code to run Lasso regression on CITE-seq data with various denoising methods applied to the RNA and ADT data.
Designed to be run on slurm cluster with wandb logging. One run for each ADT, cross-fold index and denoising method combination.

Author: Daniel Rawlinson
email: daniel.rawlinson@unimelb.edu.au
affiliations:
    Peter Doherty Institute for Infection and Immunity
    Melbourne Integrative Genomics (MIG)

'''

import argparse

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from sklearn.linear_model import LinearRegression, Lasso, LassoCV 
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

import pickle
import importlib

spec = importlib.util.spec_from_file_location("CitePred", "/home/danrawlinson/punim1597/Projects/CITE-seq/CitePred_functions.py")
CitePred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CitePred)

data_dir = '/home/danrawlinson/punim1597/Data/cite-seq_datasets/simple_data'
os.chdir('/data/projects/punim1597/Projects/CITE-seq/models_on_simple_datasets')

#argparse
adt_options = ['CD3_TotalSeqB', 'CD4_TotalSeqB', 'CD8a_TotalSeqB', 'CD14_TotalSeqB',
       'CD15_TotalSeqB', 'CD16_TotalSeqB', 'CD56_TotalSeqB', 'CD19_TotalSeqB',
       'CD25_TotalSeqB', 'CD45RA_TotalSeqB', 'CD45RO_TotalSeqB',
       'PD-1_TotalSeqB', 'TIGIT_TotalSeqB', 'CD127_TotalSeqB',
        ]
parser = argparse.ArgumentParser(description='Run Lasso regression on CITE-seq data with various denoising methods applied to the RNA and ADT data.')
parser.add_argument('--dataset', choices = ['scvi','nextgem'], help='Dataset to use', required=True)
parser.add_argument('--cellbender_rna', action='store_true', help='Use cellbender denoised RNA data')
parser.add_argument('--dsb_adt', action='store_true', help='Use DSB denoised ADT data')
parser.add_argument('--kfold', required=True, choices = list(range(0,10)), type=int, help = 'Index of kfold (0-9) to use for partition splitting')
parser.add_argument('--adt', choices = adt_options, required=True, help = 'ADT to predict')
#need to think about whether I should include a (no)scale ADT option for dsb normalised data. As per https://github.com/niaid/dsb/issues/34
args = parser.parse_args()

if args.dataset == 'scvi':
    rna_cellbender = os.path.join(data_dir, 'scvi_data', 'cellbender_out', 'cellbender_denoised_filtered.h5')
    rna = os.path.join(data_dir, 'scvi_data', 'scvi_rna.prepd.h5ad')

    adt_dsb = os.path.join(data_dir, 'scvi_data', 'dsb_out', 'scvi_data.dsb_adt.h5ad')
    adt = os.path.join(data_dir, 'scvi_data', 'scvi_adt.prepd.h5ad')

elif args.dataset == 'nextgem':
    rna_cellbender = os.path.join(data_dir, 'next_gem_10x_data', 'cellbender_out', 'cellbender_denoised_filtered.h5')
    rna = os.path.join(data_dir, 'next_gem_10x_data', 'nextgem_filtered_rna.prepd.h5ad')

    adt_dsb = os.path.join(data_dir, 'next_gem_10x_data', 'dsb_out', 'next_gem_10x_data.dsb_adt.h5ad')
    adt = os.path.join(data_dir, 'next_gem_10x_data', 'nextgem_filtered_adt.prepd.h5ad')


print(args)
if args.cellbender_rna:
    rna_data = sc.read_10x_h5(rna_cellbender)
    rna_data.var_names_make_unique() #because cellbender hasn't seen filtered rna h5ad yet
else:
    rna_data = sc.read_h5ad(rna)
if args.dsb_adt:
    adt_data = sc.read_h5ad(adt_dsb)
else:
    adt_data = sc.read_h5ad(adt)

#exit if ADT is not in common with nextgem dataset
if args.dataset == 'nextgem' and args.adt not in adt_data.var_names:
    exit('ADT not in nextgem dataset')


#prep data
sc.pp.normalize_total(rna_data,target_sum=10000)
if not args.dsb_adt: #already normalised if coming out of dsb
    adt_data = CitePred.normalize_geometric(adt_data)

#ensure same cells in both data
common_cells = list(set(rna_data.obs_names) & set(adt_data.obs_names))
rna_data = rna_data[common_cells,:]
adt_data = adt_data[common_cells,:]

sc.pp.log1p(rna_data)
sc.pp.highly_variable_genes(rna_data, n_top_genes=10000)

sc.pp.scale(rna_data)
sc.pp.scale(adt_data)

rna_data = CitePred.subset_to_highly_variable(rna_data)

#split data - have changed this to 10-fold CV
#id kfold splits

with open(f'/home/danrawlinson/punim1597/Projects/CITE-seq/models_on_simple_datasets/{args.dataset}_10fold_splits.pkl', 'rb') as file:
    kfold_dict = pickle.load(file)


#check order. Won't matter too much because cells are being called by name
CitePred.check_order(rna_data, adt_data)

#pull out train and test cell names
train_obs = [cell for cell in kfold_dict[args.kfold]['train'] if cell in common_cells]
test_obs = [cell for cell in kfold_dict[args.kfold]['test'] if cell in common_cells]

#split data
X_train, X_test = [rna_data[train_obs,:], rna_data[test_obs,:]]
y_train, y_test = [adt_data[train_obs,:], adt_data[test_obs,:]]

#keep in dict format
data_splits = {'train': [X_train, y_train], 'test': [X_test, y_test]}

print(data_splits)

#init wandb
wandb.login()

adt = args.adt

print(data_splits['train'][1][:,adt].to_df())

config = {'dataset': args.dataset,
        'model': 'LASSO',
        'ADT': adt,
        'kfold_idx': args.kfold,
        'feature_selection': '10000 highly_variable',
        'n_train_samples': data_splits['train'][0].n_obs,
        'n_test_samples': data_splits['test'][0].n_obs,
        'denoise_rna': 'cellbender' if args.cellbender_rna else None,
        'denoise_adt': 'dsb' if args.dsb_adt else None}

wandb.init(project='CITE-seq', entity = 'dnra-university-of-melbourne', config=config, group='lasso_various_denoising', tags=['full-sweep'])

#model run
lasso = LassoCV(cv=5, random_state=42, n_alphas=20,  n_jobs = -1)
lasso.fit(data_splits['train'][0].to_df(), data_splits['train'][1][:,adt].X.toarray().ravel())

train_true = data_splits['train'][1][:,adt].X.toarray().ravel()
train_preds = lasso.predict(data_splits['train'][0].to_df())
train_rmse = mean_squared_error(train_true, train_preds, squared = False)
train_R2 = lasso.score(data_splits['train'][0].to_df(), train_true)
train_pearson, _ = pearsonr(train_true, train_preds)
train_spearman, _ = spearmanr(train_true, train_preds)

test_true = data_splits['test'][1][:,adt].X.toarray().ravel()
test_preds = lasso.predict(data_splits['test'][0].to_df())
test_rmse = mean_squared_error(test_true, test_preds, squared=False)
test_R2 = lasso.score(data_splits['test'][0].to_df(), test_true)
test_pearson, _ =  pearsonr(test_true, test_preds)
test_spearman, _ = spearmanr(test_true, test_preds)

#log true and predicted values
true_and_pred_df = pd.DataFrame({'test_true': test_true, 'test_preds': test_preds})
true_and_pred_df['barcode'] = data_splits['test'][1].obs_names
#might be able to retrieve the following columns from metadata when retrieving tables, but include for now
true_and_pred_df['adt'] = adt
true_and_pred_df['kfold'] = args.kfold
true_and_pred_df['cellbender_rna'] = True if args.cellbender_rna else False
true_and_pred_df['dsb_adt'] = True if args.dsb_adt else False
true_and_pred_df['model'] = config['model']


table = wandb.Table(dataframe=true_and_pred_df)

#create and log plot
g = sns.regplot(data=true_and_pred_df , x = 'test_true', y = 'test_preds', line_kws={'color': 'red'}, scatter_kws={'s': 1})
g.set_xlabel(f'{adt} True')
g.set_ylabel(f'{adt} Predicted')

wandb.log({
        'train_R2': train_R2,
        'train_rmse' : train_rmse,
        'train_pearson': train_pearson,
        'train_spearman': train_spearman,
        'test_R2': test_R2,
        'test_rmse': test_rmse,
        'test_pearson': test_pearson,
        'test_spearman': test_spearman,
        'alpha': lasso.alpha_,
        'true_and_predicted_values': table,
        'true_vs_pred_scatter_plot': wandb.Image(g.figure),
        'training_vars': list(data_splits['train'][0].var_names)
        })

#save model
model_name = f"lasso_cv_model_{adt}_{args.kfold}_{args.cellbender_rna}_{args.dsb_adt}"
model_filename = f"{model_name}.pkl"
with open(os.path.join('lasso_models_out',model_filename), 'wb') as file:
    pickle.dump(lasso, file)

# Log the model artifact
artifact = wandb.Artifact(model_name, type='model')
artifact.add_file(os.path.join('lasso_models_out',model_filename))
wandb.log_artifact(artifact)

wandb.finish()

#delete model from local filesystem
os.remove(os.path.join('lasso_models_out',model_filename))
