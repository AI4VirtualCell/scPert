from copy import deepcopy
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.distributed as dist
from torch.nn.parallel import DataParallel

from .model import scPert_Model
from .inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis
from .utils import loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

class scPert:
    """
    scPert base model class
    """

    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'scPert', 
                 exp_name = 'scPert',
                 embedding_path=None):
        """
        Initialize scPert model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'scPert'
        exp_name: str
            Experiment name for wandb. Default: 'scPert'

        Returns
        -------
        None

        """

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}     
        self.embedding_path = embedding_path
        
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'],
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}

    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False, 
                        ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        self.config = {'hidden_size': hidden_size,
                      'embedding_size': 512,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       "pert_names": self.pert_list,
                       "pert_node_map":self.node_map_pert
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            df_jaccard = pd.read_csv(os.path.join(self.data_path, '/home/lumk/scpert/demo/data/go_essential_all.csv'))
            k=num_similar_genes_co_express_graph
            edge_list = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,['importance'])).reset_index(drop = True)
            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = scPert_Model(self.config, self.embedding_path ).to(self.device)
        self.best_model = deepcopy(self.model)
        
    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        exclude_keys = ['device', 'num_genes', 'num_perts', 'embedding_size', 'pert_names', 'pert_node_map']
        # Remove parameters that are not needed by model_initialize
        for key in exclude_keys:
            config.pop(key, None)

        # Initialize the model with the loaded configuration
        if 'uncertainty' in config:
            del config['uncertainty']
            del config['uncertainty_reg']
        self.model_initialize(**config)
        self.config = config

        # Load the model state
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location=torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            path to save the model

        Returns
        -------
        None

        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))

    def predict(self, pert_list, type_list=True, dir=None):
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome

        """
        
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']            
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        
        from torch_geometric.data import DataLoader
        
        if type_list:
            for pert in pert_list:
                if isinstance(pert, str):
                    pert_key = pert.replace('+', '_')
                    pert_for_model = pert  
                else:
                    pert_key = '_'.join(pert)
                    pert_for_model = '+'.join(pert)  
                
                try:
                    # If prediction is already saved, then skip inference
                    results_pred[pert_key] = self.saved_pred[pert_key]
                    continue
                except KeyError:
                    pass
                
                cg = create_cell_graph_dataset_for_prediction(pert_for_model, self.ctrl_adata,
                                                        self.pert_list, self.device)
                loader = DataLoader(cg, 300, shuffle=False)
                batch = next(iter(loader))
                batch.to(self.device)

                with torch.no_grad():
                    p = self.best_model(batch)
                
                all_pert_key = f"all_{pert_key}"
                results_pred[pert_key] = np.mean(p.detach().cpu().numpy(), axis=0)
                results_pred[all_pert_key] = p.detach().cpu().numpy()
        else:
            pert = pert_list
            if isinstance(pert, str):
                pert_key = pert.replace('+', '_')
                pert_for_model = pert
            else:
                pert_key = '_'.join(pert)
                pert_for_model = '+'.join(pert)
            
            try:
                # If prediction is already saved, then skip inference
                results_pred[pert_key] = self.saved_pred[pert_key]
            except KeyError:
                pass
            
            cg = create_cell_graph_dataset_for_prediction(pert_for_model, self.ctrl_adata,
                                                    self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle=False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                p = self.best_model(batch)
            
            all_pert_key = f"all_{pert_key}"
            results_pred[pert_key] = np.mean(p.detach().cpu().numpy(), axis=0)
            results_pred[all_pert_key] = p.detach().cpu().numpy()
                
        self.saved_pred.update(results_pred)

        if dir == None:
            np.savez(f"./pred_scpert_{pert_key}.npz", **results_pred)
        else:
            np.savez(f"./pred_scpert_{pert_key}.npz", **results_pred)
        return results_pred
    def plot_perturbation_heatmap(self, conditions=None, save_file=None):
        """
        Plot the perturbation comparison with distinct visualization for up/down regulation
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import matplotlib.colors as mcolors

        # Set up the plotting style
        plt.style.use('default')  # Use default style instead of 'white'
        sns.set_style("white")
        plt.rcParams['font.size'] = 12

        adata = self.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        if conditions is None:
            conditions = [c for c in adata.obs.condition.unique() if c != 'ctrl']

        all_predictions = []
        all_truth = []
        genes_list = []
        
        def calculate_logfc(treatment, control, pseudocount=1.0):
            """Calculate logFC with pseudocount"""
            treatment_mean = np.mean(treatment + pseudocount, axis=0)
            control_mean = np.mean(control + pseudocount, axis=0)
            return np.log2(treatment_mean / control_mean)
        
        # Process each condition
        for query in conditions:
            de_idx = [gene2idx[gene_raw2id[i]] for i in
                    adata.uns['top_non_dropout_de_20'][cond2name[query]]]
            genes = [gene_raw2id[i] for i in
                    adata.uns['top_non_dropout_de_20'][cond2name[query]]]
            genes_list.append(genes)
            
            treatment = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
            control = adata[adata.obs.condition == 'ctrl'].X.toarray()[:, de_idx]
            
            truth_logfc = calculate_logfc(treatment, control)
            
            query_ = query
            if isinstance(query_, str):
                pred_key = query_[0]
                pred = self.predict(query_[0])[pred_key][de_idx]
            else:
                pred_key = query_
                pred = self.predict(query_)[pred_key][de_idx]
                
            pred_logfc = calculate_logfc(
                pred.reshape(1, -1),
                control
            )
            
            all_predictions.append(pred_logfc)
            all_truth.append(truth_logfc)

        reference_genes = genes_list[0]
        pred_df = pd.DataFrame(all_predictions, 
                            index=conditions,
                            columns=reference_genes)
        truth_df = pd.DataFrame(all_truth,
                            index=conditions,
                            columns=reference_genes)

        def create_updown_colormap():
            # Create a colormap that's blue for negative, white for zero, and red for positive
            colors_down = plt.cm.Blues(np.linspace(0.2, 1, 100))
            colors_up = plt.cm.Reds(np.linspace(0.2, 1, 100))
            
            # Add white color in the middle
            white = np.array([[1, 1, 1, 1]])
            colors = np.vstack([colors_down[::-1], white, colors_up])
            return mcolors.LinearSegmentedColormap.from_list('UpDown', colors)

        # Create figure with white background
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, len(conditions)/2), facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Get maximum absolute value for symmetric scale
        max_val = max(abs(pred_df.values.max()), abs(pred_df.values.min()),
                    abs(truth_df.values.max()), abs(truth_df.values.min()))
        
        # Create custom colormap
        cmap = create_updown_colormap()
        
        # Custom normalization to make zero exactly in the middle
        norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        
        # Plot predictions
        im1 = ax1.imshow(pred_df.values, aspect='auto', cmap=cmap, norm=norm)
        ax1.set_title("Prediction", pad=15, color='black')
        ax1.set_xticks(np.arange(len(reference_genes)))
        ax1.set_yticks(np.arange(len(conditions)))
        ax1.set_xticklabels(reference_genes, rotation=45, ha='right', color='black')
        ax1.set_yticklabels(conditions, color='black')
        ax1.tick_params(axis='both', colors='black')
        
        # Plot ground truth
        im2 = ax2.imshow(truth_df.values, aspect='auto', cmap=cmap, norm=norm)
        ax2.set_title("Ground Truth", pad=15, color='black')
        ax2.set_xticks(np.arange(len(reference_genes)))
        ax2.set_yticks(np.arange(len(conditions)))
        ax2.set_xticklabels(reference_genes, rotation=45, ha='right', color='black')
        ax2.set_yticklabels(conditions, color='black')
        ax2.tick_params(axis='both', colors='black')
        
        # Add colorbars with clear up/down labels
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar2 = plt.colorbar(im2, ax=ax2)
        
        # Add labels to indicate up/down regulation
        for cbar in [cbar1, cbar2]:
            cbar.set_label('', rotation=270, labelpad=15, color='black')
            cbar.ax.tick_params(colors='black')
            cbar.ax.yaxis.label.set_color('black')
            # Remove numerical tick labels
            cbar.set_ticks([])
            # Add text annotations for up/down regulation
            cbar.ax.text(3.5, max_val * 0.75, 'Up-regulated', ha='left', va='center', color='black')
            cbar.ax.text(3.5, -max_val * 0.75, 'Down-regulated', ha='left', va='center', color='black')
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()
        
        return {
            'prediction': pred_df,
            'truth': truth_df
        }
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        """
        Predict the GI scores following perturbation of a given gene combination

        Parameters
        ----------
        combo: list
            list of genes to be perturbed
        GI_genes_file: str
            path to the file containing genes with high mean expression

        Returns
        -------
        GI scores for the given combinatorial perturbation based on scPert
        predictions

        """
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[f'{combo[0]}+ctrl']
            pred[combo[1]] = self.saved_pred[f'{combo[1]}+ctrl']
            pred['+'.join(combo)] = self.saved_pred['+'.join(combo)]
        except:
            pred = self.predict([f'{combo[0]}+ctrl', f'{combo[1]}+ctrl', f'{combo[0]}+{combo[1]}'])
        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 
        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
        pred = {p: pred[p][GI_genes_idx] 
                for p in pred 
                if "all" not in p.lower()}
        
        return get_GI_params(pred, combo)





    def plot_perturbation_heatmap_1(self, key_genes=[ 'ACTB', 'ASCL1', 'BATF3', 'BCL6', 'CDX2', 'CFLAR', 'CTNNB1', 'FGFR2', 'FGFR3', 'FOXA1', 
                                                    'HMGA2', 'HNF1A', 'HSPD1', 'JUN', 'KLB', 'LDHA', 'LMO2', 'MAF', 'MDM4', 'MEF2C', 'MYB', 
                                                    'MYH9', 'MYOG', 'NAV3', 'NEUROD1', 'NFE2L2', 'PAX5', 'PDGFRA', 'POU2F3', 'PPP1R12A', 
                                                    'PRDM1', 'RELB', 'RPL22L1', 'RUNX1', 'SHOC2', 'SLC2A1', 'SPI1', 'SYK', 'TBX2', 'TFAP2A', 
                                                    'TFAP2C', 'TFRC', 'TNFSF10', 'TUBB4B', 'UBC', 'ZBTB18', 'ZFP36L1'], 
                                conditions=None, save_file=None):
        # def plot_perturbation_heatmap(self, key_genes=['FGFR2', 'FGFR3', 'NFE2L2', 'PDGFRA', 'SYK', 'TFRC', 'TUBB4B'], conditions=None, save_file=None):
        """
        Plot the perturbation comparison for specified key genes
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import matplotlib.colors as mcolors

        # Set up the plotting style
        plt.style.use('default')
        sns.set_style("white")
        plt.rcParams['font.size'] = 12

        adata = self.adata
        gene2idx = self.node_map

        def normalize_condition_name(condition):
            """
            Normalize condition names by sorting parts split by '+' to ensure consistent naming
            """
            if '+' in condition:
                parts = condition.split('+')
                return '+'.join(sorted(parts))
            return condition
        missing_genes = [gene for gene in key_genes if gene not in gene2idx]
        if missing_genes:
            raise KeyError(f"Following genes not found in the dataset: {', '.join(missing_genes)}")
        gene_indices = [gene2idx[gene] for gene in key_genes]

        if conditions is None:
            # Get all unique conditions except 'ctrl'
            all_conditions = [c for c in adata.obs.condition.unique() if c != 'ctrl']
            # Normalize condition names
            normalized_conditions = [normalize_condition_name(c) for c in all_conditions]
            # Remove duplicates while preserving order
            conditions = list(dict.fromkeys(normalized_conditions))
        else:
            # Normalize provided conditions
            conditions = [normalize_condition_name(c) for c in conditions]
            conditions = list(dict.fromkeys(conditions))

        all_predictions = []
        all_truth = []
        
        def calculate_logfc(treatment, control, pseudocount=1.0):
            """Calculate logFC with pseudocount"""
            treatment_mean = np.mean(treatment + pseudocount, axis=0)
            control_mean = np.mean(control + pseudocount, axis=0)
            return np.log2(treatment_mean / control_mean)
        
        # Process each condition
        for query in conditions:
            # Find all original condition names that normalize to this query
            original_conditions = [c for c in adata.obs.condition.unique() 
                                if normalize_condition_name(c) == query]
            
            if not original_conditions:
                print(f"Warning: No matching conditions found for {query}")
                continue
                
            # Use the first matching condition name
            original_query = original_conditions[0]
            
            # Combine data from all matching conditions
            treatment_data = []
            for orig_cond in original_conditions:
                treatment_data.append(adata[adata.obs.condition == orig_cond].X.toarray()[:, gene_indices])
            treatment = np.vstack(treatment_data) if len(treatment_data) > 1 else treatment_data[0]
            
            control = adata[adata.obs.condition == 'ctrl'].X.toarray()[:, gene_indices]
            
            truth_logfc = calculate_logfc(treatment, control)
            
            if isinstance(original_query, str):
                pred_key = original_query
                pred = self.predict(original_query, type_list= False)[gene_indices]

                
            pred_logfc = calculate_logfc(
                pred.reshape(1, -1),
                control
            )
            
            all_predictions.append(pred_logfc)
            all_truth.append(truth_logfc)
            

        pred_df = pd.DataFrame(all_predictions, 
                            index=conditions,
                            columns=key_genes)
        truth_df = pd.DataFrame(all_truth,
                            index=conditions,
                            columns=key_genes)

        def create_updown_colormap():
            colors_down = plt.cm.Blues(np.linspace(0.2, 1, 100))
            colors_up = plt.cm.Reds(np.linspace(0.2, 1, 100))
            white = np.array([[1, 1, 1, 1]])
            colors = np.vstack([colors_down[::-1], white, colors_up])
            return mcolors.LinearSegmentedColormap.from_list('UpDown', colors)

        # Create figure with white background
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, len(conditions)/2), facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Get maximum absolute value for symmetric scale
        max_val = max(abs(pred_df.values.max()), abs(pred_df.values.min()),
                    abs(truth_df.values.max()), abs(truth_df.values.min()))
        
        # Create custom colormap
        cmap = create_updown_colormap()
        norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        
        # Plot predictions
        im1 = ax1.imshow(pred_df.values, aspect='auto', cmap=cmap, norm=norm)
        ax1.set_title("Prediction", pad=15, color='black')
        ax1.set_xticks(np.arange(len(key_genes)))
        ax1.set_yticks(np.arange(len(conditions)))
        ax1.set_xticklabels(key_genes, rotation=45, ha='right', color='black')
        ax1.set_yticklabels(conditions, color='black')
        ax1.tick_params(axis='both', colors='black')
        
        # Plot ground truth
        im2 = ax2.imshow(truth_df.values, aspect='auto', cmap=cmap, norm=norm)
        ax2.set_title("Ground Truth", pad=15, color='black')
        ax2.set_xticks(np.arange(len(key_genes)))
        ax2.set_yticks(np.arange(len(conditions)))
        ax2.set_xticklabels(key_genes, rotation=45, ha='right', color='black')
        ax2.set_yticklabels(conditions, color='black')
        ax2.tick_params(axis='both', colors='black')
        
        # Add colorbars
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar2 = plt.colorbar(im2, ax=ax2)
        
        for cbar in [cbar1, cbar2]:
            cbar.set_label('', rotation=270, labelpad=15, color='black')
            cbar.ax.tick_params(colors='black')
            cbar.ax.yaxis.label.set_color('black')
            cbar.set_ticks([])
            cbar.ax.text(3.5, max_val * 0.75, 'Up-regulated', ha='left', va='center', color='black')
            cbar.ax.text(3.5, -max_val * 0.75, 'Down-regulated', ha='left', va='center', color='black')
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()
        
        return {
            'prediction': pred_df,
            'truth': truth_df
        }
    def plot_perturbation(self, query, save_file = None):
        """
        Plot the perturbation graph

        Parameters
        ----------
        query: str
            condition to be queried
        save_file: str
            path to save the plot

        Returns
        -------
        None

        """

        import seaborn as sns
        import matplotlib.pyplot as plt
        
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        de_idx = [gene2idx[gene_raw2id[i]] for i in
                adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in
                adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        
        query_ = [q for q in query.split('+') if q != 'ctrl']
        query_str = '+'.join(query_) 
        query_key = '_'.join(query_) 
        pred_result = self.predict([query_str]) 
        pred = pred_result[query_key][de_idx]
        
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[
            de_idx].values

        pred = pred - ctrl_means
        truth = truth - ctrl_means
        
        plt.figure(figsize=[16.5,4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False,
                    medianprops = dict(linewidth=0))    

        for i in range(pred.shape[0]):
            _ = plt.scatter(i+1, pred[i], color='red')

        plt.axhline(0, linestyle="dashed", color = 'green')

        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation = 90)

        plt.ylabel("Change in Gene Expression over Control",labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        plt.show()
    
    def train(self, epochs=20, 
            lr=0.002,
            weight_decay=1e-5,
            use_parallel=True,
            device_ids=None,
            key_genes=None
            ):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """

        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)



        best_model = deepcopy(self.model)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader))
        min_val = np.inf
        if key_genes:
            gene2idx = self.node_map
            gene_indices = [gene2idx[gene] for gene in key_genes]
        else:
            gene_indices = None

        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                pred = self.model(batch)
                loss= loss_fct(pred, y, perts=batch.pert,
                                ctrl = self.ctrl_expression, 
                                dict_filter = self.dict_filter,
                            model_params=self.model.parameters(),
                            class_weights_indices=gene_indices)      
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()


                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))

            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,self.device)
            val_res = evaluate(val_loader, self.model,self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
            
            # Print epoch performance for DE genes
            log = "Train Top 20 DE MSE: {:.4f} " \
                  "Validation Top 20 DE MSE: {:.4f}. "
            print_sys(log.format(train_metrics['mse_de'],
                             val_metrics['mse_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m],
                               'train_de_' + m: train_metrics[m + '_de'],
                               'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                if isinstance(self.model, nn.DataParallel):
                    best_model = deepcopy(self.model.module)
                else:
                    best_model = deepcopy(self.model)
                # best_model = deepcopy(self.model)
                
        print_sys("Done!")
        self.best_model = best_model
        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        # Model testing
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_res = evaluate(test_loader, self.best_model,self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))
        print("test_metrics is :", test_metrics)
        print("test_pert_res is :", test_pert_res)
        
        if self.wandb:
            metrics = ['mse', 'pearson']
            for m in metrics:
                self.wandb.log({'test_' + m: test_metrics[m],
                           'test_de_'+m: test_metrics[m + '_de']                     
                          })
                
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout',
                               'frac_sigma_below_1_non_dropout',
                               'mse_top20_de_non_dropout']
        
        if self.wandb:
            for m in metrics:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

            for m in metrics_non_dropout:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

        if self.split == 'simulation':
            print_sys("Start doing subgroup analysis for simulation split...")
            subgroup = self.subgroup
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in list(list(test_pert_res.values())[0].keys()):
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m, res in test_pert_res[pert].items():
                        subgroup_analysis[name][m].append(res)

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

            ## deeper analysis
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(out[pert][m])

                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(out_non_dropout[pert][m])

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        print_sys('Done!')


