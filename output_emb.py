
# python output_emb.py --config=config_fb15k237.yaml --g_config=ComplEx --d_config=TransE
# output Entities and Relations Embeddings of ComplEx, TransE, TransD with different datasets.

import os
import logging
import datetime
import torch
from random import sample, random

from config import config, overwrite_config_with_args, dump_config
from read_data import index_ent_rel, graph_size, read_data
from data_utils import heads_tails, inplace_shuffle, batch_by_num
from trans_e import TransE
from trans_d import TransD
from distmult import DistMult
from compl_ex import ComplEx
from simple_avg import SimplE_avg
from logger_init import logger_init
from select_gpu import select_gpu
from corrupter import BernCorrupterMulti

import numpy as np


logger_init()
torch.cuda.set_device(select_gpu())
overwrite_config_with_args()
dump_config()

task_dir = config().task.dir
kb_index = index_ent_rel(os.path.join(task_dir, 'train.txt'),
                         os.path.join(task_dir, 'valid.txt'),
                         os.path.join(task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

models = {'TransE': TransE, 'TransD': TransD, 'DistMult': DistMult, 'ComplEx': ComplEx, 'SimplE_avg':SimplE_avg}
gen_config = config()[config().g_config]
dis1_config = config()[config().d_config]
dis2_config = config()["TransD"]
gen = models[config().g_config](n_ent, n_rel, gen_config)
dis1 = models[config().d_config](n_ent, n_rel, dis1_config)
dis2 = models["TransD"](n_ent, n_rel, dis2_config)
gen.load(os.path.join(task_dir, gen_config.model_file))  # load ComplEx
dis1.load(os.path.join(task_dir, dis1_config.model_file))  # load TransE
dis2.load(os.path.join(task_dir, dis2_config.model_file))  # load TransD


# ComplEx
print(gen_config.model_file)
concat_ent_emb_ComplEx=torch.cat((gen.mdl.ent_re_embed.weight,gen.mdl.ent_im_embed.weight),dim=1)
print("writting ComplEx Entity Embeddings...")
f=open(os.path.join(os.getcwd(),"pretrain_emb",config().task.dir+"ComplEx_Entity.txt"),"w")
#ent_emb_ComplEx=np.around(concat_ent_emb_ComplEx.cpu().data.numpy(),decimals=6)
ent_emb_ComplEx=concat_ent_emb_ComplEx.cpu().data.numpy()
num_ent=ent_emb_ComplEx.shape[0]
dim_ent=ent_emb_ComplEx.shape[1]
for i in range(num_ent):
    for j in range(dim_ent):
        f.write(str(ent_emb_ComplEx[i,j]))
        f.write('\t')
    f.write('\n')
f.close()
print("ComplEx Entity Embedding written")

print("writting ComplEx Relation Embeddings...")
concat_rel_emb_ComplEx=torch.cat((gen.mdl.rel_re_embed.weight,gen.mdl.rel_im_embed.weight),dim=1)
f=open(os.path.join(os.getcwd(),"pretrain_emb",config().task.dir+"ComplEx_Relation.txt"),"w")
#rel_emb_ComplEx=np.around(concat_rel_emb_ComplEx.cpu().data.numpy(),decimals=6)
rel_emb_ComplEx=concat_rel_emb_ComplEx.cpu().data.numpy()
num_rel=rel_emb_ComplEx.shape[0]
dim_rel=rel_emb_ComplEx.shape[1]
for i in range(num_rel):
    for j in range(dim_rel):
        f.write(str(rel_emb_ComplEx[i,j]))
        f.write('\t')
    f.write('\n')
f.close()
print("ComplEx Relation Embedding written")


# TransE
print(dis1_config.model_file)
print("writting TransE Entity Embeddings...")
f=open(os.path.join(os.getcwd(),"pretrain_emb",config().task.dir+"TransE_Entity.txt"),"w")
#ent_emb_TransE=np.around(dis1.mdl.ent_embed.weight.cpu().data.numpy(),decimals=6)
ent_emb_TransE=dis1.mdl.ent_embed.weight.cpu().data.numpy()
num_ent=ent_emb_TransE.shape[0]
dim_ent=ent_emb_TransE.shape[1]
for i in range(num_ent):
    for j in range(dim_ent):
        f.write(str(ent_emb_TransE[i,j]))
        f.write('\t')
    f.write('\n')
f.close()
print("TransE Entity Embedding written")

print("writting TransE Relation Embeddings...")
f=open(os.path.join(os.getcwd(),"pretrain_emb",config().task.dir+"TransE_Relation.txt"),"w")
#rel_emb_TransE=np.around(dis1.mdl.rel_embed.weight.cpu().data.numpy(),decimals=6)
rel_emb_TransE=dis1.mdl.rel_embed.weight.cpu().data.numpy()
num_rel=rel_emb_TransE.shape[0]
dim_rel=rel_emb_TransE.shape[1]
for i in range(num_rel):
    for j in range(dim_rel):
        f.write(str(rel_emb_TransE[i,j]))
        f.write('\t')
    f.write('\n')
f.close()
print("TransE Relation Embedding written")


# TransD
print(dis2_config.model_file)
print("writting TransD Entity Embeddings...")
f=open(os.path.join(os.getcwd(),"pretrain_emb",config().task.dir+"TransD_Entity.txt"),"w")
#ent_emb_TransD=np.around(dis2.mdl.proj_ent_embed.weight.cpu().data.numpy(),decimals=6)
ent_emb_TransD=dis2.mdl.proj_ent_embed.weight.cpu().data.numpy()
num_ent=ent_emb_TransD.shape[0]
dim_ent=ent_emb_TransD.shape[1]
for i in range(num_ent):
    for j in range(dim_ent):
        f.write(str(ent_emb_TransD[i,j]))
        f.write('\t')
    f.write('\n')
f.close()
print("TransD Entity Embedding written")

print("writting TransD Relation Embeddings...")
f=open(os.path.join(os.getcwd(),"pretrain_emb",config().task.dir+"TransD_Relation.txt"),"w")
#rel_emb_TransD=np.around(dis2.mdl.proj_rel_embed.weight.cpu().data.numpy(),decimals=6)
rel_emb_TransD=dis2.mdl.proj_rel_embed.weight.cpu().data.numpy()
num_rel=rel_emb_TransD.shape[0]
dim_rel=rel_emb_TransD.shape[1]
for i in range(num_rel):
    for j in range(dim_rel):
        f.write(str(rel_emb_TransD[i,j]))
        f.write('\t')
    f.write('\n')
f.close()
print("TransD Relation Embedding written")

