#!/~/.conda/envs/scvi/bin/python
'''
Python code to train Neural Networks with a GCN layer on CITE-seq data with various denoising methods applied to the RNA and ADT data.
Designed to be run on slurm cluster with wandb logging. One run for each ADT, cross-fold index and denoising method combination.
Complements the cross fold and training structure of `models_on_simple_datasets/lasso_various_denoising.py`

Author: Daniel Rawlinson
email: daniel.rawlinson@unimelb.edu.au
affiliations:
    Peter Doherty Institute for Infection and Immunity
    Melbourne Integrative Genomics (MIG)

'''
import argparse

import torch
import pandas as pd
import os
import scanpy as sp
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import seaborn as sns
import re

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import wandb
import pickle
import importlib
import sys


#geometric imports
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data as tgDat

spec = importlib.util.spec_from_file_location("CitePred", "/home/danrawlinson/punim1597/Projects/CITE-seq/CitePred_functions.py")
CitePred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CitePred)

data_dir = '/data/projects/punim1597/Data/cite-seq_datasets/simple_data/'
os.chdir('/data/projects/punim1597/Projects/CITE-seq/models_on_simple_datasets')

#argparse
adt_options = ['CD3_TotalSeqB', 'CD4_TotalSeqB', 'CD8a_TotalSeqB', 'CD14_TotalSeqB',
       'CD15_TotalSeqB', 'CD16_TotalSeqB', 'CD56_TotalSeqB', 'CD19_TotalSeqB',
       'CD25_TotalSeqB', 'CD45RA_TotalSeqB', 'CD45RO_TotalSeqB',
       'PD-1_TotalSeqB', 'TIGIT_TotalSeqB', 'CD127_TotalSeqB',
        ]
parser = argparse.ArgumentParser(description='Run NN training on CITE-seq data with various denoising methods applied to the RNA and ADT data.')
parser.add_argument('--dataset', choices = ['scvi','nextgem'], help='Dataset to use', required=True)
parser.add_argument('--cellbender_rna', action='store_true', help='Use cellbender denoised RNA data')
parser.add_argument('--dsb_adt', action='store_true', help='Use DSB denoised ADT data')
parser.add_argument('--kfold', required=True, choices = list(range(0,10)), type=int, help = 'Index of kfold (0-9) to use for partition splitting')
parser.add_argument('--adt', choices = adt_options, required=True, help = 'ADT to predict')
parser.add_argument('--batch_size', type=int, default=300, help='Batch size for training')  
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train for')
parser.add_argument('--graph_strategy', choices=['together','separate'], default='together', help='Whether to build a single graph for train and test data together or build separate graphs')
parser.add_argument('n_neighbors', type=int, default=20, help='Number of neighbors to include in the graph')
#parser.add_argument('-j', '--jaccard_adjacency_thresh', type=float, default=0.1, help='Jaccard shared-neighbhor adjacency threshold for determining edges in the graph')
#need to think about whether I should include a (no)scale ADT option for dsb normalised data. As per https://github.com/niaid/dsb/issues/34
args = parser.parse_args()

if args.dataset == 'scvi':
    rna_cellbender_path = os.path.join(data_dir, 'scvi_data', 'cellbender_out', 'cellbender_denoised_filtered.h5')
    rna_path = os.path.join(data_dir, 'scvi_data', 'scvi_rna.prepd.h5ad')

    adt_dsb_path = os.path.join(data_dir, 'scvi_data', 'dsb_out', 'scvi_data.dsb_adt.h5ad')
    adt_path = os.path.join(data_dir, 'scvi_data', 'scvi_adt.prepd.h5ad')

elif args.dataset == 'nextgem':
    rna_cellbender_path = os.path.join(data_dir, 'next_gem_10x_data', 'cellbender_out', 'cellbender_denoised_filtered.h5')
    rna_path = os.path.join(data_dir, 'next_gem_10x_data', 'nextgem_filtered_rna.prepd.h5ad')

    adt_dsb_path = os.path.join(data_dir, 'next_gem_10x_data', 'dsb_out', 'next_gem_10x_data.dsb_adt.h5ad')
    adt_path = os.path.join(data_dir, 'next_gem_10x_data', 'nextgem_filtered_adt.prepd.h5ad')



print(args)
# Load data
if args.cellbender_rna:
    rna_data = sp.read_10x_h5(rna_cellbender_path)
    rna_data.var_names_make_unique() #because cellbender hasn't seen filtered rna h5ad yet
else:
    rna_data = sp.read_h5ad(rna_path)
if args.dsb_adt:
    adt_data = sp.read_h5ad(adt_dsb_path)
else:
    adt_data = sp.read_h5ad(adt_path)

#exit if ADT is not in common with nextgem dataset
if args.dataset == 'nextgem' and args.adt not in adt_data.var_names:
    exit('ADT not in nextgem dataset')

#prep data
sp.pp.normalize_total(rna_data,target_sum=10000)
if not args.dsb_adt: #already normalised if coming out of dsb
    adt_data = CitePred.normalize_geometric(adt_data)

#ensure same cells in both data
common_cells = list(set(rna_data.obs_names) & set(adt_data.obs_names))
rna_data = rna_data[common_cells,:]
adt_data = adt_data[common_cells,:]

sp.pp.log1p(rna_data)
sp.pp.highly_variable_genes(rna_data, n_top_genes=10000)

sp.pp.scale(rna_data)
sp.pp.scale(adt_data)

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

#decide whether to graph train and test data together or separately
if args.graph_strategy == 'together':
    x_data_to_graph = [rna_data]
else:
    x_data_to_graph = [X_train, X_test]

#build edge index list for train and test X data
for adata in x_data_to_graph:
    #T cell receptor genes to remove
    import re
    tcr_match = re.compile('^(HUMAN_)?TR[AB]')
    tr_genes = []
    for i in adata.var_names:
        if tcr_match.match(i) != None: tr_genes.append(i)

    #make sure TR genes aren't marked as highly_variable for inclusion in PCA
    rna_data.var.loc[tr_genes, 'highly_variable'] = False

    #PCA
    sp.tl.pca(adata, svd_solver='arpack', )

    #Nearest Neighbour Graph. Some confusion here about wheter it uses Jaccard to incorporate shared neighbours.
    #   https://github.com/scverse/scanpy/issues/277
    sp.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=40, )  #default metric/norm is euclidean (l2)

    adj = adata.obsp['distances']
    adj[adj.nonzero()] = 1
    edge_list = torch.tensor(data = adj.A).nonzero().t().contiguous() #have checked here. each 'source' has args.n_neighbors -1 neighbors.
    adata.uns['edge_list'] = edge_list
    shared_neighbours = adj @ adj.transpose() #matrix multiply across rows of adjacency matrix
    jaccard_intersection = shared_neighbours.A * adj.A

    jaccard_union = np.full(adj.shape, 2*(args.n_neighbors -1)) #Default value is double the number of neighbors #init union #probably don't need to init a whole matrix. just init a vector of the relevant pairs?
    bool_arr = np.array(adj.toarray(), dtype='bool') #to allow OR operation 
    for source, target in zip(edge_list[0], edge_list[1]):
        jaccard_union[source, target] = np.logical_or(bool_arr[source,:], bool_arr[target,:]).sum()
    jaccard_mat = jaccard_intersection / jaccard_union
    edge_idx = np.ravel_multi_index(edge_list.numpy(), jaccard_mat.shape)
    edge_weights = jaccard_mat.take(edge_idx)
    adata.uns['edge_weights'] = edge_weights
    #thresh = args.jaccard_adjacency_thresh #threshold unused for now

#keep in dict format
data_splits = {'train': [X_train, y_train], 'test': [X_test, y_test]}
adt = args.adt
print(data_splits)
    
#Data class
# class Data(tgDat):
#     def __init__(self, X, y, edges):
#         #self.X = torch.from_numpy(X.astype(np.float32))
#         #self.y = torch.from_numpy(y.astype(np.float32)).squeeze()
#         #self.edges = edges
#         #self.len = self.X.shape[0]
#         #self.n_vars = self.X.shape[1]

#      def __getitem__(self, index):
#          return self.X[index], self.y[index]
   
#      def __len__(self):
#          return self.len
    
#dataloaders
if args.graph_strategy == 'together':
    joined_data = tgDat(x = x_data_to_graph[0].X, y = adt_data[:,adt].X, edge_index = x_data_to_graph[0].uns['edge_list'].flip([0]))
    train_loader = NeighborLoader(joined_data,
                                input_nodes=torch.tensor(np.where(np.isin(x_data_to_graph[0].obs_names, train_obs))[0]),
                                num_neighbors=[-1],
                                batch_size=args.batch_size,
                                replace=False,
                                shuffle = True)

    test_loader = NeighborLoader(joined_data,
                                input_nodes = torch.tensor(np.where(np.isin(x_data_to_graph[0].obs_names, test_obs))[0]),
                                num_neighbors=[-1], #-1 uses all neighbors
                                batch_size=x_data_to_graph[0].X.shape[0])
else:
    train_data = tgDat(x = data_splits['train'][0].X, y = data_splits['train'][1][:,adt].X, edge_index = data_splits['train'][0].uns['edge_list'].flip([0]))
    test_data = tgDat(x = data_splits['test'][0].X, y = data_splits['test'][1][:,adt].X, edge_index = data_splits['test'][0].uns['edge_list'].flip([0]))

    train_loader = NeighborLoader(train_data,
                                input_nodes=torch.tensor(range(train_data.x.shape[0])),
                                num_neighbors=[-1],
                                batch_size=args.batch_size,
                                replace=False,
                                shuffle=True)


    test_loader = NeighborLoader(test_data, 
                                input_nodes = None,
                                num_neighbors=[-1], #-1 uses all neighbors
                                batch_size=test_data.x.shape[0])

##GCN layer
class TranscriptConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = 'mean')
        #self.flow = 'target_to_source' #set to default of 'source_to_target' because edge_list has been flipped
        self.lin = nn.Linear(in_channels, out_channels, bias=False) #commented out to avoid transformation
        self.bias = nn.Parameter(torch.Tensor(out_channels)) #just tells the model that it's a parameter tensor.
        self.reset_parameters()
        
    
    def reset_parameters(self): #Called during __init__ to make sure all parameters are zeroed.
        self.lin.reset_parameters()
        self.bias.data.zero_()  
    
    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j # .view() creates a new tensor, with -1 indicating that this dimension is inferred from the input
                     
    def forward(self, x, edge_index): #input to this is the transcript x cell data itself and the edge_index

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes= x.shape[0]) #this is needed so that it's own value is included in the mean
            #self loops deactivated here to examine results of aggregation without modifying edge_list
        
              
        #degree calculation, normalization
        #N = D^(-.5)AD^(-.5) #This isn't code - just the equation written down.
        #For directed graph the above has become N = D^(-1)A
        
        row, col = edge_index 
        #row is list of edges coming in with each value an index to a node. col is list of edges going out with each value an index to a node.
        
        deg = degree(row, x.size(0), dtype=x.dtype) #This is the diagonal matrix D in the above equation. degree only calculates on 1-dimensional index
    
        #deg_inv_sqrt = deg.pow(-0.5) #element wise power. changed to .pow(-1) below for directed edges
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 #values which were 0 before being inversed put back to 0
        #when the edge is bi-direcitonal, it has to be normalised by in-degrees at target and source. for directed edge, just normalised by indegree at target.
        
        norm = deg_inv_sqrt[row] #put in same order as edge index
        #And propogate. This is where the magic happens: function calls message(), aggregate(), and update() internally
        out = self.propagate(edge_index, x=x, norm=norm) #norm is required argument. 

        out = self.lin(out) #linear transormation of input x.

        #Leave intercept in, but not sure if necessary
        out += self.bias #add bias to output
        
        return out

#init model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

n_in, n_h1, n_h2, n_h3 = rna_data.shape[1], 1000, 256, 64

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.transcript_conv1 = TranscriptConv(n_in, n_h1)
        self.fc1 = nn.Linear(n_h1, n_h2)
        self.fc2 = nn.Linear(n_h2, n_h3)
        self.fc3 = nn.Linear(n_h3, 1)
        
    def forward(self, data):
        x = F.relu(self.transcript_conv1(data.x, data.edge_index))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

model = GCN().to(device)

#init wandb
wandb.login()

config = {'dataset': args.dataset,
        'model': 'GCN',
        'layers': [n_in, n_h1, n_h2, n_h3],
        'ADT': adt,
        'kfold_idx': args.kfold,
        'feature_selection': '10000 highly_variable',
        'n_train_samples': len(train_obs),
        'n_test_samples': len(test_obs),
        'denoise_rna': 'cellbender' if args.cellbender_rna else None,
        'denoise_adt': 'dsb' if args.dsb_adt else None,
        'graph_strategy': args.graph_strategy,
        'n_neighbors': args.n_neighbors}

wandb.init(project='CITE-seq', entity='dnra-university-of-melbourne', group='GCN_various_denoising', config=config, tags = 
           ['full-sweep','one-hop', 'graphing-together'])


loss_fn = nn.MSELoss()

#number of epochs
epochs = args.epochs

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #this is the default lr anyway

log_freq = 5
wandb.watch(model, log_freq=log_freq) #log gradients as model trains every 50 steps(batches)
wandb.config.update({"epochs": epochs, "log_freq": log_freq, "batch_size": train_loader.batch_size, "loss_fn": loss_fn.__class__.__name__, "optimizer": optimizer.__class__.__name__})

#train model
total_batches = 0 #for logging

for e in range(epochs):
    for idx, data in enumerate(train_loader):
        #data.x incorporates the target nodes, but also all the nodes feeding into it. That's why X.shape[1] != batch_size
        #batcher actually batches the edge_index tensor so that n_target_nodes == batch_size. try it with: len(data.edge_index[1].unique())

        #data.edge_index now has 19 edges for each unique target. This is because the edge_index has been flipped 

        y = data.y[:data.batch_size].to(device) #so we have to mask the non-target nodes
        #zero gradients
        optimizer.zero_grad()
        pred = model(data.to(device))
        pred = pred[:data.batch_size] #trim to batch node predictions
        loss = loss_fn(pred.squeeze(),y.squeeze())    
        loss.backward()
        optimizer.step()

        #increment total batches
        total_batches += 1

        if total_batches % log_freq == 0:
            #calc test loss at each update point
            with torch.no_grad():
                for _, data in enumerate(test_loader): #just one in this iterator
                    y = data.y[:data.batch_size].to(device) #trim to batch nodes #not really necessary when using all nodes in test test_loader
                    test_pred = model(data.to(device))
                    test_pred = test_pred[:data.batch_size] #trim to batch nodes
                    test_loss = loss_fn(test_pred.squeeze(),y.squeeze())
            #log metrics
            wandb.log({'train_loss': loss.item(), 'test_loss': test_loss.item(),'epoch': e, 'batch': idx})
            sys.stdout.write(f'Epoch {e}, Batch {idx}, Train Loss {loss.item()}, Test Loss {test_loss.item()}\n')
            sys.stdout.flush()


#evaluate model on all samples
#model.eval() #don't need this because I have no dropout and no normalisation layers

#train preds
if args.graph_strategy == 'together':
    all_train_loader = NeighborLoader(joined_data, input_nodes=torch.tensor(np.where(np.isin(x_data_to_graph[0].obs_names, train_obs))[0]), 
                                      num_neighbors=[-1], batch_size=len(train_obs))
else:
    all_train_loader = NeighborLoader(train_data, input_nodes=list(range(0,len(train_obs))), num_neighbors=[-1], batch_size=len(train_obs))
    
with torch.no_grad():
    for _, data in enumerate(all_train_loader):
        train_pred = model(data.to(device))[:data.batch_size] #trim to batch nodes

#test preds
with torch.no_grad():
    for _, data in enumerate(test_loader):
        test_pred = model(data.to(device))[:data.batch_size] #trim to batch nodes


#log summary metrics
train_true = adt_data[train_obs,adt].X.toarray().ravel()
train_preds = train_pred.squeeze().cpu().numpy()
train_rmse = mean_squared_error(train_true, train_preds, squared = False)
train_R2 = r2_score(train_true , train_preds)
train_pearson, _ = pearsonr(train_true, train_preds)
train_spearman, _ = spearmanr(train_true, train_preds)

test_true = adt_data[test_obs,adt].X.toarray().ravel()
test_preds = test_pred.squeeze().cpu().numpy()
test_rmse = mean_squared_error(test_true, test_preds, squared=False)
test_R2 = r2_score(test_true, test_preds)
test_pearson, _ =  pearsonr(test_true, test_preds)
test_spearman, _ = spearmanr(test_true, test_preds)

#log true and predicted values
true_and_pred_df = pd.DataFrame({'test_true': test_true, 'test_preds': test_preds})
true_and_pred_df['barcode'] = test_obs
#might be able to retrieve the following columns from metadata when retrieving tables, but include for now
true_and_pred_df['adt'] = adt
true_and_pred_df['kfold'] = args.kfold
true_and_pred_df['cellbender_rna'] = True if args.cellbender_rna else False
true_and_pred_df['dsb_adt'] = True if args.dsb_adt else False
true_and_pred_df['model'] = config['model']
true_and_pred_df['dataset'] = args.dataset

#table for logging
table = wandb.Table(dataframe=true_and_pred_df)

#create and log plot
g = sns.jointplot(data=true_and_pred_df , x = 'test_true', y = 'test_preds', 
                  kind = 'reg', line_kws={'color': 'red'}, scatter_kws={'s': 1},
                  truncate = False)
g.set_axis_labels(f'{adt} True', f'{adt} Predicted')

wandb.log({'train_RMSE': train_rmse,
            'train_R2': train_R2,
            'train_pearson': train_pearson,
            'train_spearman': train_spearman,
            'test_RMSE': test_rmse,
            'test_R2': test_R2,
            'test_pearson': test_pearson,
            'test_spearman': test_spearman,
            'true_and_predicted_values': table,
            'true_vs_pred_scatter_plot': wandb.Image(g.figure),
            'training_vars': list(rna_data.var_names)
            })

# Not logging models. Takes too much space to store all at this stage
# model_name = f"GCN_model_{adt}_{args.kfold}_{args.cellbender_rna}_{args.dsb_adt}"
# model_filename = f"{model_name}.pkl"
# with open(os.path.join('GCN_models_out',model_filename), 'wb') as file:
#     pickle.dump(model, file)

# Log the model artifact
# artifact = wandb.Artifact(model_name, type='model')
# artifact.add_file(os.path.join('GCN_models_out',model_filename))
# wandb.log_artifact(artifact)

wandb.finish()
#delete model from local filesystem
# os.remove(os.path.join('GCN_models_out',model_filename))