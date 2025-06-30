import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle
import sys, os
import requests
from torch_geometric.data import Data
from zipfile import ZipFile
import tarfile
from sklearn.linear_model import TheilSenRegressor
from dcor import distance_correlation
from multiprocessing import Pool

def parse_single_pert(i):
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return pert

def parse_combo_pert(i):
    return i.split('+')[0], i.split('+')[1]

def combine_res(res_1, res_2):
    res_out = {}
    for key in res_1:
        res_out[key] = np.concatenate([res_1[key], res_2[key]])
    return res_out

def parse_any_pert(p):
    if ('ctrl' in p) and (p != 'ctrl'):
        return [parse_single_pert(p)]
    elif 'ctrl' not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]

def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    
    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        print_sys("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        
def zip_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for zip file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset
    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.zip')
        print_sys('Extracting zip file...')
        with ZipFile((save_path + '.zip'), 'r') as zip:
            zip.extractall(path = data_path)
        print_sys("Done!")  
        
def tar_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for tar file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset

    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.tar.gz')
        print_sys('Extracting tar file...')
        with tarfile.open(save_path  + '.tar.gz') as tar:
            tar.extractall(path= data_path)
        print_sys("Done!")  
        
def get_go_auto(gene_list, data_path, data_name):
    """
    Get gene ontology data

    Args:
        gene_list (list): list of gene names
        data_path (str): the path to save the extracted dataset
        data_name (str): the name of the dataset

    Returns:
        df_edge_list (pd.DataFrame): gene ontology edge list
    """
    go_path = os.path.join(data_path, data_name, 'go.csv')
    
    if os.path.exists(go_path):
        return pd.read_csv(go_path)
    else:
        ## download gene2go.pkl
        if not os.path.exists(os.path.join(data_path, 'gene2go.pkl')):
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
            dataverse_download(server_path, os.path.join(data_path, 'gene2go.pkl'))
        with open(os.path.join(data_path, 'gene2go.pkl'), 'rb') as f:
            gene2go = pickle.load(f)

        gene2go = {i: list(gene2go[i]) for i in gene_list if i in gene2go}
        edge_list = []
        for g1 in tqdm(gene2go.keys()):
            for g2 in gene2go.keys():
                edge_list.append((g1, g2, len(np.intersect1d(gene2go[g1],
                   gene2go[g2]))/len(np.union1d(gene2go[g1], gene2go[g2]))))

        edge_list_filter = [i for i in edge_list if i[2] > 0]
        further_filter = [i for i in edge_list if i[2] > 0.1]
        df_edge_list = pd.DataFrame(further_filter).rename(columns = {0: 'gene1',
                                                                      1: 'gene2',
                                                                      2: 'score'})

        df_edge_list = df_edge_list.rename(columns = {'gene1': 'source',
                                                      'gene2': 'target',
                                                      'score': 'importance'})
        df_edge_list.to_csv(go_path, index = False)        
        return df_edge_list

class GeneSimNetwork():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """
    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """

        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        
        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                      self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        #self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
        
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list
        
def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    fname = './data/go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list}

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                      total=len(gene2go.keys())))
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})
    
    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list

def get_similarity_network(network_type, adata, threshold, k,
                           data_path, data_name, split, seed, train_gene_set_size,
                           set2conditions, default_pert_graph=True, pert_list=None):
    
    if network_type == 'co-express':
        df_out = get_coexpression_network_from_train(adata, threshold, k,
                                                     data_path, data_name, split,
                                                     seed, train_gene_set_size,
                                                     set2conditions)
    elif network_type == 'go':
        if default_pert_graph:
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
            tar_data_download_wrapper(server_path, 
                                     os.path.join(data_path, 'go_essential_all'),
                                     data_path)
            df_jaccard = pd.read_csv(os.path.join(data_path, 
                                     'go_essential_all/go_essential_all.csv'))

        else:
            df_jaccard = make_GO(data_path, pert_list, data_name)

        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,
                                    ['importance'])).reset_index(drop = True)

    return df_out

def get_coexpression_network_from_train(adata, threshold, k, data_path,
                                        data_name, split, seed, train_gene_set_size,
                                        set2conditions):
    """
    Infer co-expression network from training data

    Args:
        adata (anndata.AnnData): anndata object
        threshold (float): threshold for co-expression
        k (int): number of edges to keep
        data_path (str): path to data
        data_name (str): name of dataset
        split (str): split of dataset
        seed (int): seed for random number generator
        train_gene_set_size (int): size of training gene set
        set2conditions (dict): dictionary of perturbations to conditions
    """
    
    fname = os.path.join(os.path.join(data_path, data_name), split + '_'  +
                         str(seed) + '_' + str(train_gene_set_size) + '_' +
                         str(threshold) + '_' + str(k) +
                         '_co_expression_network.csv')
    
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        gene_list = [f for f in adata.var.gene_name.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list)) 
        X = adata.X
        train_perts = set2conditions['train']
        X_tr = X[np.isin(adata.obs.condition, [i for i in train_perts if 'ctrl' in i])]
        gene_list = adata.var['gene_name'].values

        X_tr = X_tr.toarray()
        out = np_pearson_cor(X_tr, X_tr)
        out[np.isnan(out)] = 0
        out = np.abs(out)

        out_sort_idx = np.argsort(out)[:, -(k + 1):]
        out_sort_val = np.sort(out)[:, -(k + 1):]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]
        df_co_expression = pd.DataFrame(df_g).rename(columns = {0: 'source',
                                                                1: 'target',
                                                                2: 'importance'})
        df_co_expression.to_csv(fname, index = False)
        return df_co_expression
    
def filter_pert_in_go(condition, pert_names):
    """
    Filter perturbations in GO graph

    Args:
        condition (str): whether condition is 'ctrl' or not
        pert_names (list): list of perturbations
    """

    if condition == 'ctrl':
        return True
    else:
        cond1 = condition.split('+')[0]
        cond2 = condition.split('+')[1]
        num_ctrl = (cond1 == 'ctrl') + (cond2 == 'ctrl')
        num_in_perts = (cond1 in pert_names) + (cond2 in pert_names)
        if num_ctrl + num_in_perts == 2:
            return True
        else:
            return False
        



# def loss_fct(pred, y, perts, ctrl = None, direction_lambda = 1e-3, dict_filter = None):
#     """
#     Main MSE Loss function, includes direction loss

#     Args:
#         pred (torch.tensor): predicted values
#         y (torch.tensor): true values
#         perts (list): list of perturbations
#         ctrl (str): control perturbation
#         direction_lambda (float): direction loss weight hyperparameter
#         dict_filter (dict): dictionary of perturbations to conditions

#     """
#     gamma = 2
#     mse_p = torch.nn.MSELoss()
#     perts = np.array(perts)
#     losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

#     for p in set(perts):
#         pert_idx = np.where(perts == p)[0]
        
#         # during training, we remove the all zero genes into calculation of loss.
#         # this gives a cleaner direction loss. empirically, the performance stays the same.
#         if p!= 'ctrl':
#             retain_idx = dict_filter[p]
#             pred_p = pred[pert_idx][:, retain_idx]
#             y_p = y[pert_idx][:, retain_idx]
#         else:
#             pred_p = pred[pert_idx]
#             y_p = y[pert_idx]
#         losses = losses + torch.sum((pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
                         
#         ## direction loss
#         if (p!= 'ctrl'):
#             losses = losses + torch.sum(direction_lambda *
#                                 (torch.sign(y_p - ctrl[retain_idx]) -
#                                  torch.sign(pred_p - ctrl[retain_idx]))**2)/\
#                                  pred_p.shape[0]/pred_p.shape[1]
#         else:
#             losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl) -
#                                                 torch.sign(pred_p - ctrl))**2)/\
#                                                 pred_p.shape[0]/pred_p.shape[1]
#     return losses/(len(set(perts)))


# Focal MSE Loss


def loss_fct(pred, y, perts, ctrl=None, direction_lambda=1e-3, dict_filter=None, 
                      l1_lambda=1e-5, cosine_lambda=0.1, focal_gamma=2, class_weights_indices=None,
                      model_params=None):
    """
    Improved Loss function for gene perturbation prediction
    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions
        l1_lambda (float): L1 regularization weight
        cosine_lambda (float): cosine similarity loss weight
        focal_gamma (float): focal loss gamma parameter
        class_weights (torch.tensor): weights for each class (gene) for weighted MSE
        model_params (list): list of model parameters for L1 regularization (optional)
    """
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    

    cos_similarity = torch.nn.CosineSimilarity(dim=1)
    
    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        
        if p != 'ctrl' and dict_filter is not None:
            retain_idx = dict_filter[p]
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = y[pert_idx][:, retain_idx]
            ctrl_p = ctrl[retain_idx] 
        else:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
            ctrl_p = ctrl
        
        # Focal MSE Loss
        focal_mse = focal_mse_loss(pred_p, y_p, ctrl_p.to(pred_p.device))
        losses = losses + focal_mse
        
        # Weighted MSE Loss (if class weights are provided)
        if class_weights_indices is not None:

            class_weights = calculate_class_weights(y,
                                                    important_genes_indices=class_weights_indices,
                                                    boost_factor=5,
                                                    base_weight=1.0,
                                                    normalize=True  # 10
                                                )
            if p != 'ctrl' and dict_filter is not None:
                weighted_mse = torch.mean(class_weights[retain_idx] * (pred_p - y_p)**2)
            else:
                weighted_mse = torch.mean(class_weights * (pred_p - y_p)**2)
            losses = losses + weighted_mse
        
        # Direction Loss (improved)
        if ctrl is not None:
            if p != 'ctrl' and dict_filter is not None:
                direction_loss = torch.mean(torch.abs(torch.sign(y_p - ctrl[retain_idx]) - 
                                                      torch.sign(pred_p - ctrl[retain_idx])))
            else:
                direction_loss = torch.mean(torch.abs(torch.sign(y_p - ctrl) - 
                                                      torch.sign(pred_p - ctrl)))
            losses = losses + direction_lambda * direction_loss
        
        # Cosine Similarity Loss
        cosine_loss = 1 - cos_similarity(pred_p.view(pred_p.size(0), -1), 
                                         y_p.view(y_p.size(0), -1)).mean()
        losses = losses + cosine_lambda * cosine_loss
    
    # L1 Regularization (if model parameters are provided)
    if model_params is not None and l1_lambda > 0:
        l1_reg = torch.tensor(0., requires_grad=True).to(pred.device)
        for param in model_params:
            l1_reg = l1_reg + torch.norm(param, 1)
        losses = losses + l1_lambda * l1_reg
    
    return losses / len(set(perts))
def focal_mse_loss(pred, target, ctrl, gamma=2):

    diff_magnitude = torch.abs(target - ctrl)
    focal_weight = torch.sigmoid(diff_magnitude)  
    mse = (pred - target)**2
    return torch.mean(mse * focal_weight**gamma)

# def loss_fct_with_tracking(pred, y, perts, ctrl=None, direction_lambda=1e-3, dict_filter=None, 
#                           l1_lambda=1e-5, cosine_lambda=0.1, focal_gamma=2, class_weights_indices=None,
#                           model_params=None, return_components=False):
#     """
#     带组件跟踪的损失函数
#     """
#     perts = np.array(perts)
#     losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    
#     # 用于跟踪各组件
#     components = {
#         'focal_mse': torch.tensor(0.0).to(pred.device),
#         'weighted_mse': torch.tensor(0.0).to(pred.device),
#         'direction_loss': torch.tensor(0.0).to(pred.device),
#         'cosine_loss': torch.tensor(0.0).to(pred.device),
#         'l1_reg': torch.tensor(0.0).to(pred.device)
#     }
    
#     cos_similarity = torch.nn.CosineSimilarity(dim=1)
#     n_perts = len(set(perts))
    
#     for p in set(perts):
#         pert_idx = np.where(perts == p)[0]
        
#         if p != 'ctrl' and dict_filter is not None:
#             retain_idx = dict_filter[p]
#             pred_p = pred[pert_idx][:, retain_idx]
#             y_p = y[pert_idx][:, retain_idx]
#             ctrl_p = ctrl[retain_idx] 
#         else:
#             pred_p = pred[pert_idx]
#             y_p = y[pert_idx]
#             ctrl_p = ctrl
        
#         # Focal MSE Loss
#         focal_mse = focal_mse_loss(pred_p, y_p, ctrl_p.to(pred_p.device))
#         components['focal_mse'] += focal_mse
#         losses = losses + focal_mse
        
#         # Weighted MSE Loss (if class weights are provided)
#         if class_weights_indices is not None:
#             class_weights = calculate_class_weights(y,
#                                                     important_genes_indices=class_weights_indices,
#                                                     boost_factor=5,
#                                                     base_weight=1.0,
#                                                     normalize=True)
#             if p != 'ctrl' and dict_filter is not None:
#                 weighted_mse = torch.mean(class_weights[retain_idx] * (pred_p - y_p)**2)
#             else:
#                 weighted_mse = torch.mean(class_weights * (pred_p - y_p)**2)
#             components['weighted_mse'] += weighted_mse
#             losses = losses + weighted_mse
        
#         # Direction Loss (improved)
#         if ctrl is not None:
#             if p != 'ctrl' and dict_filter is not None:
#                 direction_loss = torch.mean(torch.abs(torch.sign(y_p - ctrl[retain_idx]) - 
#                                                       torch.sign(pred_p - ctrl[retain_idx])))
#             else:
#                 direction_loss = torch.mean(torch.abs(torch.sign(y_p - ctrl) - 
#                                                       torch.sign(pred_p - ctrl)))
#             components['direction_loss'] += direction_lambda * direction_loss
#             losses = losses + direction_lambda * direction_loss
        
#         # Cosine Similarity Loss
#         cosine_loss = 1 - cos_similarity(pred_p.view(pred_p.size(0), -1), 
#                                          y_p.view(y_p.size(0), -1)).mean()
#         components['cosine_loss'] += cosine_lambda * cosine_loss
#         losses = losses + cosine_lambda * cosine_loss
    
#     # L1 Regularization (if model parameters are provided)
#     if model_params is not None and l1_lambda > 0:
#         l1_reg = torch.tensor(0., requires_grad=True).to(pred.device)
#         for param in model_params:
#             l1_reg = l1_reg + torch.norm(param, 1)
#         components['l1_reg'] = l1_lambda * l1_reg
#         losses = losses + l1_lambda * l1_reg
    
#     # 平均化组件 
#     for key in components:
#         components[key] = components[key] / n_perts
    
#     total_loss = losses / n_perts
#     components['total_loss'] = total_loss
    
#     if return_components:
#         return total_loss, components
#     else:
#         return total_loss

def calculate_class_weights(y, 
                           important_genes_indices,
                           base_weight=1.0,
                           boost_factor=10,
                           normalize=True):
    """
     Weight calculation functions that directly augment the weights of specified genes
    
    Args:
        y (torch.Tensor):  [n_samples, n_genes]
        important_genes_indices (list): index of important genes
        base_weight (float): Base weight value (shared by all genes by default)
        boost_factor (float): Weight enhancement multiplier for important genes
        normalize (bool): Whether to normalise so that the weights are averaged to 1
    """
    # 初始化基础权重
    weights = torch.full((y.shape[1],), base_weight, dtype=torch.float32)
    
    # 增强重要基因权重
    if important_genes_indices:
        # 过滤无效索引
        valid_indices = [idx for idx in important_genes_indices if idx < y.shape[1]]
        weights[valid_indices] *= boost_factor
    
    # 可选归一化（保持权重均值不变）
    if normalize:
        weights = weights / weights.mean()
    
    return weights.to(y.device)



# # Function to get gene index
# def get_gene_index(gene_name, adata):
#     """Get the index of a gene in the dataset"""
#     if 'gene_name' in adata.var.columns:
#         gene_names = pd.Index(adata.var.gene_name)
#         if gene_name in gene_names:
#             return gene_names.get_loc(gene_name)
    
#     # If gene_name column doesn't exist or gene not found there, try with index
#     if gene_name in adata.var_names:
#         return adata.var_names.get_loc(gene_name)
    
#     # Try alternative names
#     if gene_name == 'CD274':
#         alternatives = ['PD-L1', 'PDL1', 'pdl1', 'B7-H1']
#         for alt in alternatives:
#             if alt in adata.var_names:
#                 return adata.var_names.get_loc(alt)
    
#     print(f"Warning: Gene {gene_name} not found in dataset")
#     return None


# Helper function to calculate class weights


# def calculate_class_weights(y, method='balanced', important_genes_indices=None, boost_factor=10):
#     """
#     Calculate class weights with boosted weights for important genes
#     """
#     gene_freq = torch.sum(y != 0, dim=0)
#     if method == 'inverse':
#         weights = 1 / (gene_freq + 1)
#     elif method == 'balanced':
#         weights = 1 / (2 * gene_freq)
#         weights[gene_freq == 0] = 1
#     else:
#         raise ValueError("Invalid method. Choose 'inverse' or 'balanced'.")
    
#     # Normalize weights
#     weights = weights / weights.sum() * len(weights)
    
#     # Boost important genes' weights
#     if important_genes_indices is not None:
#         weights[important_genes_indices] *= boost_factor
    
#     return weights


def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush = True, file = sys.stderr)
    
def create_cell_graph_for_prediction(X, pert_idx, pert_gene):
    """
    Create a perturbation specific cell graph for inference

    Args:
        X (np.array): gene expression matrix
        pert_idx (list): list of perturbation indices
        pert_gene (list): list of perturbations

    """

    # if pert_idx is None:
    #     pert_idx = [-1]
    return Data(x=torch.Tensor(X).T, pert_idx = None, pert=pert_gene)
    

def create_cell_graph_dataset_for_prediction(pert_gene, ctrl_adata, gene_names,
                                             device, num_samples = 300):
    """
    Create a perturbation specific cell graph dataset for inference

    Args:
        pert_gene (list): list of perturbations
        ctrl_adata (anndata): control anndata
        gene_names (list): list of gene names
        device (torch.device): device to use
        num_samples (int): number of samples to use for inference (default: 300)

    """

    # Get the indices (and signs) of applied perturbation
    # pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene]

    Xs = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :].X.toarray()
    # Create cell graphs
    cell_graphs = [create_cell_graph_for_prediction(X, None, pert_gene).to(device) for X in Xs]
    return cell_graphs

##
##GI related utils
##

def get_coeffs(singles_expr, first_expr, second_expr, double_expr):
    """
    Get coefficients for GI calculation

    Args:
        singles_expr (np.array): single perturbation expression
        first_expr (np.array): first perturbation expression
        second_expr (np.array): second perturbation expression
        double_expr (np.array): double perturbation expression

    """
    results = {}
    results['ts'] = TheilSenRegressor(fit_intercept=False,
                          max_subpopulation=1e5,
                          max_iter=1000,
                          random_state=1000)   
    X = singles_expr
    y = double_expr
    results['ts'].fit(X, y.ravel())
    Zts = results['ts'].predict(X)
    results['c1'] = results['ts'].coef_[0]
    results['c2'] = results['ts'].coef_[1]
    results['mag'] = np.sqrt((results['c1']**2 + results['c2']**2))
    
    results['dcor'] = distance_correlation(singles_expr, double_expr)
    results['dcor_singles'] = distance_correlation(first_expr, second_expr)
    results['dcor_first'] = distance_correlation(first_expr, double_expr)
    results['dcor_second'] = distance_correlation(second_expr, double_expr)
    results['corr_fit'] = np.corrcoef(Zts.flatten(), double_expr.flatten())[0,1]
    results['dominance'] = np.abs(np.log10(results['c1']/results['c2']))
    results['eq_contr'] = np.min([results['dcor_first'], results['dcor_second']])/\
                        np.max([results['dcor_first'], results['dcor_second']])
    
    return results

def get_GI_params(preds, combo):
    """
    Get GI parameters

    Args:
        preds (dict): dictionary of predictions
        combo (list): list of perturbations

    """
    singles_expr = np.array([preds[f'{combo[0]}+ctrl'], preds[f'{combo[1]}+ctrl']]).T
    first_expr = np.array(preds[f'{combo[0]}+ctrl']).T
    second_expr = np.array(preds[f'{combo[1]}+ctrl']).T
    double_expr = np.array(preds[f'{combo[0]}+{combo[1]}']).T
    #之前是"_"
    
    # print(singles_expr,double_expr)
    return get_coeffs(singles_expr, first_expr, second_expr, double_expr)

def get_GI_genes_idx(adata, GI_gene_file):
    """
    Optional: Reads a file containing a list of GI genes (usually those
    with high mean expression)

    Args:
        adata (anndata): anndata object
        GI_gene_file (str): file containing GI genes (generally corresponds
        to genes with high mean expression)
    """
    # Genes used for linear model fitting
    GI_genes = np.load(GI_gene_file, allow_pickle=True)
    # GI_genes = GI_gene_file
    GI_genes_idx = np.where([g in GI_genes for g in adata.var.gene_name.values])[0]
    
    return GI_genes_idx

def get_mean_control(adata):
    """
    Get mean control expression
    """
    mean_ctrl_exp = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()
    return mean_ctrl_exp

def get_genes_from_perts(perts):
    """
    Returns list of genes involved in a given perturbation list
    """

    if type(perts) is str:
        perts = [perts]
    gene_list = [p.split('+') for p in np.unique(perts)]
    gene_list = [item for sublist in gene_list for item in sublist]
    gene_list = [g for g in gene_list if g != 'ctrl']
    return list(np.unique(gene_list))

def filter_gi_score_data(adata):
    # 确保 'condition' 列在 adata.obs 中
    if 'condition' not in adata.obs.columns:
        raise ValueError("'condition' column not found in adata.obs")
    
    # 创建一个字典来存储每个基因的扰动情况
    gene_perturbations = {}
    
    # 遍历所有条件
    for condition in adata.obs['condition']:
        genes = condition.split('+')
        if condition == 'ctrl':
            # 保存对照组
            gene_perturbations.setdefault('ctrl', set()).add(condition)
        # elif len(genes) == 1:
        #     # 保存单个基因扰动
        #     gene_perturbations.setdefault(genes[0], set()).add(condition)
        # elif len(genes) == 2:
        else:
            if 'ctrl' not in genes:
                # 这是一个基因组合 (a+b)
                gene_perturbations.setdefault(genes[0], set()).add(condition)
                gene_perturbations.setdefault(genes[1], set()).add(condition)
            else:
                # 这是一个单基因扰动 (a+ctrl 或 ctrl+a)
                gene = next(gene for gene in genes if gene != 'ctrl')
                gene_perturbations.setdefault(gene, set()).add(condition)
    
    # 找出所有满足条件的扰动组合
    valid_perturbations = set()
    valid_perturbations.add('ctrl')  # 添加对照组
    
    for gene, perturbations in gene_perturbations.items():
        if gene != 'ctrl':
            # 添加单个基因扰动，包括 "gene+ctrl" 和 "ctrl+gene" 的形式
            single_pert = next((p for p in perturbations if len(p.split('+')) == 1 or 'ctrl' in p.split('+')), None)
            if single_pert:
                valid_perturbations.add(single_pert)
            
            # 添加基因组合
            combined_perts = [p for p in perturbations if '+' in p and 'ctrl' not in p]
            for combined_pert in combined_perts:
                other_gene = next(g for g in combined_pert.split('+') if g != gene)
                other_single_pert = next((p for p in gene_perturbations.get(other_gene, []) if len(p.split('+')) == 1 or 'ctrl' in p.split('+')), None)
                if other_single_pert:
                    valid_perturbations.update([single_pert, other_single_pert, combined_pert])
    
    # 创建一个布尔索引来筛选数据
    valid_index = adata.obs['condition'].isin(valid_perturbations)
    
    # 使用布尔索引来筛选 AnnData 对象
    filtered_adata = adata[valid_index].copy()
    
    return filtered_adata