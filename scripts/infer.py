import sys
import torch
sys.path.append(r'./')
from models import ProcePertdata,scpert
import pandas as pd
import numpy as np
from multiprocessing import Pool
import tqdm
import scanpy as sc

# 强制设置默认的CUDA设备为cuda:1
torch.cuda.set_device('cuda:2')

model_name = 'norman'
model_path = r'/home/lumk/scpert/demo/norman_model_FINAL'
data_path = r"/home/lumk/scpert/demo/data"
pertData = ProcePertdata.PertData(data_path)
pertData.load(DataName = 'norman')
pertData.prepare_split(split = 'no_test', seed = 77)
pertData.get_dataloader(batch_size = 128, test_batch_size = 128)

# 显式指定device='cuda:1'，并确保模型在此设备上
SCPert = scpert.scPert(pertData, device = 'cuda:2', 
                    weight_bias_track = False, 
                    proj_name = 'norman', 
                    exp_name = model_name,
                    embedding_path="/home/lumk/scpert/scGPT/embeddings/gene_embeddings_norman_512.npy")



SCPert.load_pretrained(model_path)


pert_file = pd.read_csv('/home/lumk/scpert/analy_pic/mid-result/gene_pert.csv')
pert_file = pert_file.fillna('ctrl')

pert_list = []
for _, row in pert_file.iterrows():
    g1, g2 = row['gene1'], row['gene2']

    pert_list.append(f"{g1}+{g2}")
    break

with torch.cuda.device('cuda:2'):
    prediction = SCPert.predict(pert_list)
    