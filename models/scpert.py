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
                       'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       "pert_names": self.pert_list,
                       "pert_node_map":self.node_map_pert
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
            
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
                
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout',
                               'frac_sigma_below_1_non_dropout',
                               'mse_top20_de_non_dropout']       

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


