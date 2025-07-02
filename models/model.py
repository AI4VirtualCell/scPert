import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='./train.log',  
    filemode='w' 
)


class MLP(nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if batch_norm and i < len(sizes) - 2:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
            elif last_layer_act == "ReLU":
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MemoryEfficientMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, chunk_size=256):
        super().__init__()
        # assert d_model % nhead == 0, 
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.chunk_size = chunk_size
        

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        return x.view(-1, batch_size, self.nhead, self.d_k).permute(1, 2, 0, 3)

    def forward(self, query, key, value, key_padding_mask=None):
        # (seq, batch, d_model)
        batch_size = query.size(1)
        

        q = self.split_heads(self.q_proj(query), batch_size)  
        k = self.split_heads(self.k_proj(key), batch_size)
        v = self.split_heads(self.v_proj(value), batch_size)
        outputs = []
        for i in range(0, query.size(0), self.chunk_size):
            q_chunk = q[:, :, i:i+self.chunk_size, :]         
            k_chunk = k[:, :, i:i+self.chunk_size, :]         
            v_chunk = v[:, :, i:i+self.chunk_size, :]        
            

            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            chunk_output = torch.matmul(attn_probs, v_chunk)
            outputs.append(chunk_output)

        output = torch.cat(outputs, dim=2)                    
        output = output.permute(2, 0, 1, 3).reshape(-1, batch_size, self.nhead * self.d_k)
        output = self.out_proj(output)                       
        return output, None  

class ImprovedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = MemoryEfficientMultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src_permuted = src.permute(1, 0, 2)  # (seq, batch, dim)
        attn_output, _ = self.self_attn(src_permuted, src_permuted, src_permuted)
        attn_output = attn_output.permute(1, 0, 2)  # (batch, seq, dim)
        
 
        src = self.norm1(src + self.dropout(attn_output))
        
     
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src


class scPert_Model(nn.Module):
    def __init__(self, args,embedding_path):
        super(scPert_Model, self).__init__()
        self.args = args
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        self.hidden_size = args['hidden_size']
        self.embedding_size = args['embedding_size']
        self.device = args['device']

        # Load gene embeddings 
        gene_data = np.load("/home/lumk/scPert/embeddings/gene_2_kge_comgcn_final_common.npz")
        self.loaded_gene_names = gene_data['gene_names']
        self.loaded_embeddings = torch.tensor(gene_data['embeddings'], dtype=torch.float32, device=self.device)
        self.gene_to_index = {gene: idx for idx, gene in enumerate(self.loaded_gene_names)}

        # Load pert embeddings
        pert_data = np.load("/home/lumk/scPert/embeddings/gene_embeddingss_full_common.npz")
        self.pert_gene_names = pert_data['gene_names']
        self.pert_embeddings = torch.tensor(pert_data['embeddings'], dtype=torch.float32, device=self.device)
        self.pert_to_index = {gene: idx for idx, gene in enumerate(self.pert_gene_names)}
        self.gene_emb = torch.tensor(np.load(embedding_path), 
                            dtype=torch.float32, device=self.device)


        self.gene_interaction_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.gene_dense = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        

        self.pert_dense = nn.Sequential(
            nn.Linear(640, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        

        self.pert_fuse = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )


        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerLayer(self.hidden_size, nhead=8, dropout=0.1)
            for _ in range(8) 
        ])
        

        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.LayerNorm(self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        self.output_layer = nn.Linear(self.hidden_size, 1)

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)


        self.random_embeddings = {}
        

        self.emb_scale = nn.Parameter(torch.ones(1))
        

        self.pert_emb = nn.Embedding(self.num_perts, self.hidden_size)
        self.control_emb = nn.Parameter(torch.randn(1, self.hidden_size) / np.sqrt(self.hidden_size))
        self.pos_emb = nn.Embedding(self.num_genes, self.hidden_size)    
        self.to(self.device)

    def get_random_embedding(self, gene_name):

        if gene_name not in self.random_embeddings:
            logging.info(f"Generating random embedding for {gene_name}")
            torch.manual_seed(hash(gene_name) % (2**32)) 
            self.random_embeddings[gene_name] = torch.randn(640, device=self.device) / np.sqrt(640)
        return self.random_embeddings[gene_name]


    def get_gene_embedding(self, gene_name):

        if gene_name == 'ctrl':
            return torch.zeros(128, device=self.device) 
        if gene_name in self.gene_to_index:
            # logging.info(f"get KGE {gene_name}")
            return self.loaded_embeddings[self.gene_to_index[gene_name]]
        return self.get_random_embedding(gene_name)[:128]

    def get_pert_embedding(self, pert_name):


        if pert_name == 'ctrl':
            return torch.zeros(512, device=self.device)  
            

        if pert_name in self.pert_to_index:
            return self.pert_embeddings[self.pert_to_index[pert_name]]
        

        return self.get_random_embedding(pert_name)[128:]

    def forward(self, pert_data):
        x, pert = pert_data.x, pert_data.pert
        num_graphs = len(pert_data.batch.unique())

        gene_emb = self.gene_emb
        gene_emb = gene_emb.repeat(num_graphs, 1)
        gene_emb = self.gene_dense(gene_emb)
        gene_emb = gene_emb.view(num_graphs, self.num_genes, -1)

        gene_interaction = torch.matmul(gene_emb, gene_emb.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        gene_interaction = F.softmax(gene_interaction, dim=-1)
        gene_context = torch.matmul(gene_interaction, gene_emb)
        gene_context = self.gene_interaction_layer(gene_context) 
        gene_emb = gene_emb + gene_context

        pert_emb_matrix = torch.zeros(num_graphs, self.num_genes, self.hidden_size, device=self.device)

        for idx, p in enumerate(pert):


            if p == 'ctrl':

                pert_emb_matrix[idx] = self.control_emb.expand(self.num_genes, -1)
                continue

            
            pert_list = p.split('+')
            pert_embs = []
            
            for pert_name in pert_list:
                new_pert_emb = self.get_gene_embedding(pert_name)
                scGPT_pert_emb = self.get_pert_embedding(pert_name)
                combined_pert_emb = torch.cat([new_pert_emb, scGPT_pert_emb])
                combined_pert_emb = self.pert_dense(combined_pert_emb)
                pert_embs.append(combined_pert_emb)

            if pert_embs:

                if len(pert_embs) == 1:
                    pert_embs = pert_embs * 2
                    
                stacked_embs = torch.stack(pert_embs)
                emb_total = self.pert_fuse(stacked_embs).mean(0, keepdim=True)
                
                if len(emb_total.shape) == 2:
                    emb_total = emb_total.mean(0, keepdim=True)
                    
                pert_emb_matrix[idx] = emb_total.repeat(self.num_genes, 1)

        control_mask = (pert_emb_matrix.sum(dim=-1) == 0).unsqueeze(-1)
        pert_emb_matrix = torch.where(control_mask, self.control_emb.expand_as(pert_emb_matrix), pert_emb_matrix)


        pert_emb_matrix = pert_emb_matrix * self.emb_scale


        combined_emb = self.layer_norm1(gene_emb + pert_emb_matrix)
        combined_emb = self.dropout(combined_emb)


        for layer in self.transformer_layers:
            layer_output = layer(combined_emb)
            combined_emb = combined_emb + layer_output * 0.5

        combined_emb = self.layer_norm2(combined_emb)

        batch_size, seq_len, hidden_size = combined_emb.shape
        output = self.final_mlp(combined_emb.view(-1, hidden_size)).view(batch_size, seq_len, hidden_size)
        output = self.output_layer(output)


        output = output.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
        output = torch.split(torch.flatten(output), self.num_genes)

        return torch.stack(output)
    
