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


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=128, type=int, help='number of users in a testing batch')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=32, type=int, help='embedding size')
    parser.add_argument('--intentNum', default=48, type=int, help='number of intents')
    parser.add_argument('--neg_num', default=60, type=int, help='number of negative nodes')
    parser.add_argument('--topk', default=[20, 40, 50, 60, 70, 80, 90], type=int, help='K of top K')
    parser.add_argument('--keepRate', default=0.9, type=float, help='ratio of edges to keep')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature')
    parser.add_argument('--MiddleIntentnum', default=32, type=int, help='intents number on mlp')
    parser.add_argument('--FinedintentNum', default=80, type=float, help='intents number after mlp')
    parser.add_argument('--intent_dim', default=16, type=float, help='intents number after mlp')
    parser.add_argument('--data', default='amazon', type=str, help='name of dataset')
    parser.add_argument('--gpu', default='2', type=str, help='indicates which gpu to use')
    return parser.parse_args(args=[])

args = ParseArgs()


class DataHandler:
    def __init__(self):
        if args.data == 'amazon':
            predir = 'Data/amazon-book/'
        elif args.data == 'ml10m':
            predir = 'Data/ml10m/'
        elif args.data == 'tmall':
            predir = 'Data/tmall/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def LoadData(self):
        with open(self.tstfile, 'rb') as fs:
            self.test_csr = (pickle.load(fs) != 0).astype(np.float32)
            
        with open(self.trnfile, 'rb') as fs:
            self.train_csr = (pickle.load(fs) != 0).astype(np.float32)
        
        args.user, args.item = self.train_csr.shape
        
        if type(self.train_csr) != coo_matrix:
            trnMat = sp.coo_matrix(self.train_csr)

        self.torchBiAdj = self.makeTorchAdj(trnMat)

        self.trnLoader = DataLoader(
            TrainDataset(self.train_csr, args.neg_num), 
            batch_size = args.batch,
            shuffle = True, 
            num_workers = 0,
            collate_fn = TrainDataset.collate_fn
        )
        

class TrainDataset(Dataset):
    def __init__(self, csr_train, n_negs):
        self.num_edge = csr_train.nnz
        self.num_U, self.num_V = csr_train.shape
        self.num_negs = n_negs
        self.src, self.dst = csr_train.nonzero()
        self.src_torch = torch.from_numpy(self.src)
        self.dst_torch = torch.from_numpy(self.dst)
        # self.csr_all = csr_all
        
        
    def __len__(self):
        return self.num_edge
    
    def __getitem__(self, idx):
        neg_idx_V = None
        if self.num_negs > 0:
            neg_idx_V = torch.randint(0, self.num_V, (self.num_negs,))    # 随机选择负样本
        return self.src_torch[idx], self.dst_torch[idx], neg_idx_V
    
    @staticmethod
    def collate_fn(data):
        idx_U = torch.stack([_[0] for _ in data], dim=0)
        pos_idx_V = torch.stack([_[1] for _ in data], dim=0)
        if data[0][2] is not None:
            neg_idx_V = torch.stack([_[2] for _ in data], dim=0)
        else:
            neg_idx_V = None
        return idx_U, pos_idx_V, neg_idx_V


def SpAdjDropEdge(adj, keepRate=1.0):
    if keepRate == 1.0:
        return adj
    vals = adj._values()
    idxs = adj._indices()
    edgeNum = vals.size()
    mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
    newVals = vals[mask] / keepRate
    newIdxs = idxs[:, mask]
    return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0.0)

    def forward(self, input):
        return self.layer(input)


class MLP(nn.Module):
    def __init__(self, dims, act='relu', dropout=0.6):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(FC(dims[i - 1], dims[i]))
        self.act = getattr(F, act)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input):
        curr_input = input
        for i in range(len(self.layers) - 1):
            hidden = self.layers[i](curr_input)
            hidden = self.act(hidden)
            if self.dropout:
                hidden = self.dropout(hidden)
            curr_input = hidden
        output = self.layers[-1](curr_input)
        return torch.sigmoid(output)

############### copy from "Billion-Scale Bipartite Graph Embedding: A Global-Local Induced Approach"

def precision_recall(r, k, n_ground_truth):
    right_pred = r[:, :k].sum(1)  # (batch, )
    n_ground_truth_denomitor = n_ground_truth.clone()
    n_ground_truth_denomitor[n_ground_truth_denomitor == 0] = 1
    batch_recall = (right_pred / n_ground_truth_denomitor).sum()
    batch_precision = right_pred.sum() / k
    return batch_recall


def ndcg(r, k, n_ground_truth):
    pred_data = r[:, :k]
    device = pred_data.device
    max_r = (torch.arange(k, device=device).expand_as(pred_data) < n_ground_truth.view(-1, 1)).float()  # (batch, k)
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2, device=device)), dim=1)  # (batch, ) as a denominator
    dcg = torch.sum(pred_data * (1. / torch.log2(torch.arange(2, k + 2, device=device))), dim=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    batch_ndcg = ndcg.sum()
    return batch_ndcg


def test_minibatch(csr_test, csr_train, test_batch):
    num_U = len(csr_test.indptr) - 1
    for begin in range(0, num_U, test_batch):
        head = csr_test.indptr[begin: min(begin + test_batch, num_U)]
        tail = csr_test.indptr[1 + begin: 1 + begin + test_batch]
        num_pos_V = tail - head
        # print('[', begin, begin + test_batch, ']', 'pos item cnt:', num_pos_V)
        # print('sum of n_items:', num_pos_V.sum())
        ground_truth = csr_test.indices[head[0]: tail[-1]]
        
        # assert num_pos_V.sum() == len(ground_truth)  # debug
        
        # print('data:', '(', len(ground_truth), ')', ground_truth)
        
        # exclude items in training set
        head_train = csr_train.indptr[begin: min(begin + test_batch, num_U)]
        tail_train = csr_train.indptr[1 + begin: 1 + begin + test_batch]
        num_V_to_exclude = tail_train - head_train
        V_to_exclude = csr_train.indices[head_train[0]: tail_train[-1]]
        
        # assert num_V_to_exclude.sum() == len(V_to_exclude)  # debug
        
        batch_size = len(num_pos_V)
        yield np.arange(begin, begin + batch_size), num_pos_V, ground_truth, num_V_to_exclude, V_to_exclude


def print_metrics(metrics, topk, max_K, print_max_K=True):
    if print_max_K:
        k = max_K
        # print(f'precision@{k}:', metrics[f'precision@{k}'], end='\t')
        print(f'recall@{k}:', metrics[f'recall@{k}'], end='\t')
        print(f'ndcg@{k}:', metrics[f'ndcg@{k}'])
    else:
        for i, k in enumerate(topk):
            # if i > 0:
            #     print('--')
            # print(f'precision@{k}:', metrics[f'precision@{k}'], end='\t')
            print(f'recall@{k}:', metrics[f'recall@{k}'], end='\t')
            print(f'ndcg@{k}:', metrics[f'ndcg@{k}'], end='\t')
            print()


def ranking_edges(TEST_LHS, TEST_LHS_INI, Batch_U_soft_assignments, Batch_U_soft_assignments_ini, RHS_EMBEDDINGS, RHS_EMBEDDINGS_INI, RHS_INTENTS, RHS_INTENTS_INI, eval_model):
    eval_model.eval()
    
    with torch.no_grad(): 

        user_pair_intent = Batch_U_soft_assignments * RHS_INTENTS_INI
        user_pair_intent = eval_model.USER_INTENT_NETWORK(user_pair_intent)

        item_pair_intent = Batch_U_soft_assignments_ini * RHS_INTENTS
        item_pair_intent = eval_model.ITEM_INTENT_NETWORK(item_pair_intent)

        SOCIAL_LINK_PROB_1 = torch.cosine_similarity(TEST_LHS.unsqueeze(1), RHS_EMBEDDINGS_INI, dim=2) + 1.0

        SOCIAL_LINK_PROB_2 = torch.cosine_similarity(TEST_LHS_INI.unsqueeze(1), RHS_EMBEDDINGS, dim=2) + 1.0

        LINK_PROB = torch.cat([user_pair_intent, item_pair_intent, SOCIAL_LINK_PROB_1.unsqueeze(-1), SOCIAL_LINK_PROB_2.unsqueeze(-1)], dim=-1)
        LINK_PROB = eval_model.LINK_NETWORK(LINK_PROB)

        return LINK_PROB.squeeze(-1)


def batch_evaluation(trained_model, csr_test, csr_train, train_adj, epoch, test_batch, topk, max_K, device='cuda:0'):

    trained_model.eval()

    U_EMDS, I_EMDS, I_EMDS_INI, I_soft_assignments, I_soft_assignments_ini = trained_model.eval_calitemclusters(train_adj)
    
    num_test_U = 0
    
    metrics = {}
    for k in topk:
        metrics[f'epoch'] = epoch
        # metrics[f'precision@{k}'] = 0.
        metrics[f'recall@{k}'] = 0.
        metrics[f'ndcg@{k}'] = 0.
    
    with tqdm(total=csr_test.shape[0], desc=f'eval epoch {epoch}') as pbar:
        for i, batch in enumerate(test_minibatch(csr_test, csr_train, test_batch)):
            # print('-' * 20)
            # print('batch', i)
            idx_U, n_ground_truth, ground_truth, num_V_to_exclude, V_to_exclude = batch
            assert idx_U.shape == n_ground_truth.shape
            assert idx_U.shape == num_V_to_exclude.shape
            # print(idx_U.shape, n_ground_truth.shape, ground_truth.shape)

            batch_size = idx_U.shape[0]
            num_U_to_exclude = (n_ground_truth == 0).sum()  # exclude users that are not in test set
            # print('num_U_to_exclude:', num_U_to_exclude)
            num_test_U += batch_size - num_U_to_exclude
            
            # -> cuda 
            idx_U = torch.tensor(idx_U, dtype=torch.long, device=device)
            n_ground_truth = torch.tensor(n_ground_truth, dtype=torch.long, device=device)
            ground_truth = torch.tensor(ground_truth, dtype=torch.long, device=device)
            num_V_to_exclude = torch.tensor(num_V_to_exclude, dtype=torch.long, device=device)
            V_to_exclude = torch.tensor(V_to_exclude, dtype=torch.long, device=device)
            
            ########################################
            # metrics calculation
            
            with torch.no_grad():

                test_lhs = U_EMDS[idx_U]

                test_lhs_ini, test_lhs_cluster, test_lhs_cluster_ini = trained_model.eval_caluserclusters(test_lhs, idx_U)

                rating = ranking_edges(test_lhs, test_lhs_ini, test_lhs_cluster, test_lhs_cluster_ini, I_EMDS, I_EMDS_INI, I_soft_assignments, I_soft_assignments_ini, trained_model)   # (args.batch, args.items)

                row_index = torch.arange(batch_size, device=device)  # (batch, )
                
                # filter out the items in the training set
                row_index_to_exclude = row_index.repeat_interleave(num_V_to_exclude)
                rating[row_index_to_exclude, V_to_exclude] = -1e6
                
                # pick the top max_K items
                _, rating_K = torch.topk(rating, k=max_K)  # rating_K: (batch, max_K)
                
                # build a test_graph based on ground truth coordinates
                row_index_ground_truth = row_index.repeat_interleave(n_ground_truth)
                test_g = torch.sparse_coo_tensor(indices=torch.stack((row_index_ground_truth, ground_truth), dim=0), values=torch.ones_like(ground_truth), 
                        size=(batch_size, args.item))
                
                # build a pred_graph based on top max_K predictions
                pred_row = row_index.repeat_interleave(max_K)
                pred_col = rating_K.flatten()
                pred_g = torch.sparse_coo_tensor(indices=torch.stack((pred_row, pred_col), dim=0), values=torch.ones_like(pred_col), size=(batch_size, args.item))
                
                # build a hit_graph based on the intersection of test_graph and pred_graph
                dense_g = (test_g * pred_g).coalesce().to_dense().float()

                r = dense_g[pred_row, pred_col].view(batch_size, -1)  # (batch, max_K)
            
                # recall, precision, ndcg
                for k in topk:
                    # recall, precision
                    # batch_recall, batch_precision = precision_recall(r, k, n_ground_truth)
                    batch_recall = precision_recall(r, k, n_ground_truth)
                    # ndcg
                    batch_ndcg = ndcg(r, k, n_ground_truth)
                    
                    # print(f'batch_precision@{k}:', batch_precision.item())
                    # print(f'batch_recall@{k}:', batch_recall.item())
                    # print(f'batch_ndcg@{k}:', batch_ndcg.item())
                    # print('--')
                    
                    # metrics[f'precision@{k}'] += batch_precision.item()
                    metrics[f'recall@{k}'] += batch_recall.item()
                    metrics[f'ndcg@{k}'] += batch_ndcg.item()
                    
            pbar.update(batch_size)
                
    for k in topk:
        # metrics[f'precision@{k}'] /= num_test_U
        metrics[f'recall@{k}'] /= num_test_U
        metrics[f'ndcg@{k}'] /= num_test_U
            
    return metrics
