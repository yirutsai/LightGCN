# from torch_geometric.datasets import MovieLens
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-11/"
from dataset import MovieLens
import torch
import torch.nn.functional as F
# from torch_geometric.nn.models import LightGCN
from model import LightGCN, LightGCN2
from torch_sparse import SparseTensor, matmul
from sklearn.model_selection import train_test_split
from utils import sample_mini_batch, bpr_loss, evaluation
# import random
from tqdm import tqdm
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

myseed = 3030
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
print("using :",device)
from torch.utils.cpp_extension import CUDA_HOME
print("CUDA_HOME=",CUDA_HOME)

# dataset = MovieLens(root = "~/data/MovieLens",rating_threshold=4,dataset_name="ml-latest-small")
dataset = MovieLens(root = "~/data/MovieLens",rating_threshold=4,dataset_name="ml-1m")
# print(vars(dataset[0]))
num_users = len(dataset.user_mapping)
num_movies = len(dataset.movie_mapping)
num_edges = len(dataset.edge_index[1])
print(f"{num_users=}")
print(f"{num_movies=}")
print(f"{max(dataset.edge_index[0])=}")
print(f"{max(dataset.edge_index[1])=}")
print(f"{num_edges=}")



all_index = [i for i in range(num_edges)]
train_index, test_index = train_test_split(all_index,test_size=0.2,random_state = myseed)
train_edge_index = dataset.edge_index[:, train_index]
test_edge_index = dataset.edge_index[:, test_index]

train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(num_users + num_movies, num_users + num_movies)).to(device)
test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(num_users + num_movies, num_users + num_movies)).to(device)

embedding_dim = 64
num_layers = 4
n_epoch = 10000
batch_size = 512
lr = 1e-3
weight_decay = 0
lamda = 1e-5
eval_steps = 200
K = 50

model = LightGCN2(num_users, num_movies, embedding_dim=embedding_dim, num_layers=num_layers, add_self_loops=False)


model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay =weight_decay)

train_losses = []
val_losses = []
for epoch in tqdm(range(1,n_epoch+1)):
    model.train()
    optimizer.zero_grad()
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(train_sparse_edge_index)

    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(batch_size,train_edge_index)
    user_indices = user_indices.to(device)
    pos_item_indices = pos_item_indices.to(device)
    neg_item_indices = neg_item_indices.to(device)

    users_emb_final = users_emb_final[user_indices]
    users_emb_0 = users_emb_0[user_indices]
    pos_items_emb_final = items_emb_final[pos_item_indices]
    pos_items_emb_0 = items_emb_0[pos_item_indices]
    neg_items_emb_final = items_emb_final[neg_item_indices]
    neg_items_emb_0 = items_emb_0[neg_item_indices]

    train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lamda)
    train_loss.backward()
    optimizer.step()

    if(epoch%eval_steps==0):
        model.eval()
        
        val_loss, recall, precision, ndcg = evaluation(model, test_edge_index, test_sparse_edge_index, [train_edge_index], K, lamda)
        print(f"[Iteraion {epoch:05d}/{n_epoch}] train_loss: {train_loss.item():.5f}, val_loss: {val_loss:.5f}, val_recall@{K}: {recall:.5f}, val_precision@{K}: {precision:.5f}, val_ndcg@{K}: {ndcg:.5f}")#
