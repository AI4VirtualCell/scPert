import os
import sys
import torch
import numpy as np
sys.path.append(r'./')
from models import ProcePertdata,scpert


torch.cuda.set_device('cuda:1')


embedding_dir = "/home/lumk/scpert/scGPT/embeddings/"

# Get all embedding files
embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]

# Base data path
data_path = "/home/lumk/scpert/demo/data"
embedding_to_data_map = {"gene_embeddings_norman_512.npy": "norman"}

# Process each dataset
for embedding_file, DataName in embedding_to_data_map.items():
    print(f"\n===== Processing dataset: {DataName} with embedding: {embedding_file} =====\n")
    
    # Initialize and prepare data
    pertData = ProcePertdata.PertData(data_path)
    pertData.load(DataName=DataName)
    pertData.prepare_split(split='simulation', seed=77)
    pertData.get_dataloader(batch_size=64, test_batch_size=64)
    embedding_path = os.path.join(embedding_dir, embedding_file)
    # Initialize scpert model
    SCPert = scpert.scPert(pertData, device='cuda:1',
                         weight_bias_track=False,
                         proj_name='pertnet',
                         exp_name=f'pertnet_{DataName}',
                         embedding_path=embedding_path)
    
    # Override the gene_emb attribute with the correct embedding file
    
    SCPert.model_initialize(hidden_size=64)
    
    # Load the correct embedding file
    
    
    # Train the model
    # SCPert.train(epochs=25, lr=0.001)
    SCPert.train(epochs=5, lr=0.001)
    # Save the model
    SCPert.save_model(f'{DataName}_model_FINAL')
    
    print(f"\n===== Completed dataset: {DataName} =====\n")

print("All datasets processed successfully!")