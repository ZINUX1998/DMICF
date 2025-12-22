import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import argparse
from torch.utils.data import Dataset, DataLoader
import os
import torch
from statistics import mean
from torch import nn
import torch.nn.functional as F
import sys
import time
import pickle
import pandas as pd
from tqdm import tqdm
import random
from sklearn import metrics
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support
from torch.nn import Parameter
from argparse import ArgumentParser
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, coo_matrix, vstack
from scipy.spatial.distance import pdist
from scipy import sparse
from torch.nn import Linear

def write_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

dataset = "ml10m"
trnfile = "Data/" + dataset + "/trnMat.pkl"
with open(trnfile, 'rb') as fs:
    train_csr = (pickle.load(fs) != 0).astype(np.float32)


user_num, item_num = train_csr.shape

if type(train_csr) != coo_matrix:
    trnMat = sp.coo_matrix(train_csr)

uu_save_path = "Data/" + dataset + "/uu_graph.pkl"

def build_user_user_graph_gpu(trnMat, ui_k, sim_threshold, save_path, block_size=500, device="cuda"):
    """
    GPU 加速版用户-用户相似度图构建 (Jaccard)
    输入:
        trnMat: scipy.sparse CSR 用户-物品交互矩阵
        ui_k: 每个用户保留的 top-K 相似用户
        sim_threshold: 相似度阈值
        block_size: 分块大小
        save_path: 保存路径
        device: "cuda" 或 "cpu"
    """

    R = trnMat.tocoo()
    n_users, n_items = R.shape

    # 转成 torch sparse (用户-物品矩阵)
    indices = torch.tensor([R.row, R.col], dtype=torch.long, device=device)
    values = torch.tensor(R.data, dtype=torch.float32, device=device)
    R_torch = torch.sparse_coo_tensor(indices, values, size=(n_users, n_items)).coalesce()

    # 每个用户的度（行和）
    user_deg = torch.tensor(np.array(trnMat.sum(axis=1)).flatten(), dtype=torch.float32, device=device)

    final_row, final_col, final_val = [], [], []

    for start in tqdm(range(0, n_users, block_size), desc="build user-user graph (GPU)"):
        end = min(start + block_size, n_users)

        # 当前 block (dense，放在 GPU)
        rows = torch.arange(start, end, device=device)
        R_block = R_torch.index_select(0, rows)  # (block_size, n_items)

        # 共现矩阵 block: (block_size, n_users)
        C_block = torch.sparse.mm(R_block, R_torch.transpose(0, 1))  

        # 遍历每个用户
        for bi in range(end - start):
            i = start + bi

            row = C_block[bi].to_dense()  # 取出 dense 一行 (可优化为 topk sparse)
            inters = row.nonzero().squeeze()
            commons = row[inters]

            if inters.numel() == 0:
                continue

            unions = user_deg[i] + user_deg[inters] - commons
            sims = commons / unions

            # 排序
            sims_sorted, idx_sorted = torch.sort(sims, descending=True)
            neighs_sorted = inters[idx_sorted]

            # top-K
            for j, s in zip(neighs_sorted[:ui_k].tolist(), sims_sorted[:ui_k].tolist()):
                if i == j: 
                    continue
                final_row.append(i)
                final_col.append(j)
                final_val.append(s)

            # 超过阈值的
            for j, s in zip(neighs_sorted.tolist(), sims_sorted.tolist()):
                if s > sim_threshold and j not in neighs_sorted[:ui_k].tolist():
                    final_row.append(i)
                    final_col.append(j)
                    final_val.append(s)

    uu_mat = {'row': final_row, 'col': final_col, 'data': final_val}
    write_pkl(save_path, uu_mat)
    print(f"User–user similarity graph saved to {save_path}")

build_user_user_graph_gpu(trnMat, ui_k=10, sim_threshold=0.8, save_path=uu_save_path)
