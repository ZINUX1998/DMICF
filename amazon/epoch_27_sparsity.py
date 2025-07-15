from evaluation import *
from train import Model
import numpy as np
import torch
from scipy.sparse import csr_matrix

handler = DataHandler()
handler.LoadData()

print('USER', args.user, 'ITEM', args.item)
print('NUM OF INTERACTIONS', handler.trnLoader.dataset.__len__())

adj = handler.torchBiAdj
train_csr = handler.train_csr
test_csr = handler.test_csr

all_csr = train_csr + test_csr

# 用户交互次数
user_interactions = all_csr.sum(axis=1).A1  # .A1 变成一维数组

# 定义区间
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
user_groups = {}

# 闭区间处理：[a, b)
for i in range(len(bins) - 1):
    low, high = bins[i], bins[i + 1]
    mask = (user_interactions >= low) & (user_interactions < high)
    indices = np.where(mask)[0]
    user_groups[f"[{low},{high})"] = indices

# 最后一个开区间：[90,+)
mask = user_interactions >= 90
indices = np.where(mask)[0]
user_groups["[90,+)"] = indices

# 保存用户分组索引
for label, indices in user_groups.items():
    print(f"{label}: {len(indices)} users")
    filename = f"user_indices_{label.replace('+', 'plus').replace(',', '_').replace('[', '').replace(')', '')}.npy"
    np.save(filename, indices)

# 加载训练好的模型
epoch = 27
trained_model = Model().cuda()
checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
trained_model.load_state_dict(checkpoint)
trained_model.eval()

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


def test_minibatch_group(csr_test, csr_train, test_batch, group_user_indices):
    """
    csr_test, csr_train: scipy.sparse csr_matrix, 是全图的测试/训练邻接矩阵
    test_batch: int, batch大小
    group_user_indices: np.array，分组内的全局用户索引
    """
    num_U = len(group_user_indices)
    for begin in range(0, num_U, test_batch):
        batch_users = group_user_indices[begin: begin + test_batch]

        head = csr_test.indptr[batch_users]
        tail = csr_test.indptr[batch_users + 1]
        num_pos_V = tail - head
        ground_truth = []
        for h, t in zip(head, tail):
            ground_truth.extend(csr_test.indices[h:t])
        ground_truth = np.array(ground_truth)

        head_train = csr_train.indptr[batch_users]
        tail_train = csr_train.indptr[batch_users + 1]
        num_V_to_exclude = tail_train - head_train
        V_to_exclude = []
        for h, t in zip(head_train, tail_train):
            V_to_exclude.extend(csr_train.indices[h:t])
        V_to_exclude = np.array(V_to_exclude)

        yield batch_users, num_pos_V, ground_truth, num_V_to_exclude, V_to_exclude


# ----------- 分组 batch 评价函数 -----------

def batch_evaluation_group(trained_model, csr_test, csr_train, train_adj,
                           epoch, test_batch, topk, max_K, group_user_indices, device='cuda:0'):
    trained_model.eval()

    # 获取全图嵌入
    U_EMDS, I_EMDS, I_EMDS_INI, I_soft_assignments, I_soft_assignments_ini = trained_model.eval_calitemclusters(train_adj)

    num_test_U = 0
    metrics = {}
    for k in topk:
        metrics[f'epoch'] = epoch
        metrics[f'recall@{k}'] = 0.
        metrics[f'ndcg@{k}'] = 0.

    with tqdm(total=len(group_user_indices), desc=f'eval epoch {epoch}') as pbar:
        for i, batch in enumerate(test_minibatch_group(csr_test, csr_train, test_batch, group_user_indices)):
            idx_U, n_ground_truth, ground_truth, num_V_to_exclude, V_to_exclude = batch

            batch_size = idx_U.shape[0]
            num_U_to_exclude = (n_ground_truth == 0).sum()
            num_test_U += batch_size - num_U_to_exclude

            idx_U = torch.tensor(idx_U, dtype=torch.long, device=device)
            n_ground_truth = torch.tensor(n_ground_truth, dtype=torch.long, device=device)
            ground_truth = torch.tensor(ground_truth, dtype=torch.long, device=device)
            num_V_to_exclude = torch.tensor(num_V_to_exclude, dtype=torch.long, device=device)
            V_to_exclude = torch.tensor(V_to_exclude, dtype=torch.long, device=device)

            with torch.no_grad():
                # 关键：使用全图嵌入索引分组用户
                test_lhs = U_EMDS[idx_U]

                test_lhs_ini, test_lhs_cluster, test_lhs_cluster_ini = trained_model.eval_caluserclusters(test_lhs, idx_U)

                rating = ranking_edges(test_lhs, test_lhs_ini, test_lhs_cluster, test_lhs_cluster_ini,
                                       I_EMDS, I_EMDS_INI, I_soft_assignments, I_soft_assignments_ini, trained_model)

                row_index = torch.arange(batch_size, device=device)

                # 排除训练集中的物品
                row_index_to_exclude = row_index.repeat_interleave(num_V_to_exclude)
                rating[row_index_to_exclude, V_to_exclude] = -1e6

                _, rating_K = torch.topk(rating, k=max_K)

                row_index_ground_truth = row_index.repeat_interleave(n_ground_truth)
                test_g = torch.sparse_coo_tensor(
                    indices=torch.stack((row_index_ground_truth, ground_truth), dim=0),
                    values=torch.ones_like(ground_truth),
                    size=(batch_size, rating.shape[1])
                )

                pred_row = row_index.repeat_interleave(max_K)
                pred_col = rating_K.flatten()
                pred_g = torch.sparse_coo_tensor(
                    indices=torch.stack((pred_row, pred_col), dim=0),
                    values=torch.ones_like(pred_col),
                    size=(batch_size, rating.shape[1])
                )

                dense_g = (test_g * pred_g).coalesce().to_dense().float()
                r = dense_g[pred_row, pred_col].view(batch_size, -1)

                for k in topk:
                    batch_recall = precision_recall(r, k, n_ground_truth)
                    batch_ndcg = ndcg(r, k, n_ground_truth)

                    metrics[f'recall@{k}'] += batch_recall.item()
                    metrics[f'ndcg@{k}'] += batch_ndcg.item()

            pbar.update(batch_size)

    for k in topk:
        metrics[f'recall@{k}'] /= num_test_U
        metrics[f'ndcg@{k}'] /= num_test_U

    return metrics


if __name__ == '__main__':
    test_batch = 128
    topk = [40]
    max_K = 40
    device = 'cuda:0'

    for label, user_indices in user_groups.items():
        print(f"Evaluating group {label}, size: {len(user_indices)}")
        metrics = batch_evaluation_group(
            trained_model=trained_model,
            csr_test=test_csr,
            csr_train=train_csr,
            train_adj=adj,
            epoch=epoch,
            test_batch=test_batch,
            topk=topk,
            max_K=max_K,
            group_user_indices=user_indices,
            device=device,
        )
        print(f"{label} recall@40: {metrics['recall@40']:.4f}")