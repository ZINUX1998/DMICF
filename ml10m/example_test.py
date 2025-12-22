############### 计算用户的hit@20
# from evaluation import * 
# from train import Model
# handler = DataHandler()
# handler.LoadData()
# print('USER', args.user, 'ITEM', args.item)
# print('NUM OF INTERACTIONS', handler.trnLoader.dataset.__len__())
# adj = handler.torchBiAdj
# train_csr = handler.train_csr
# test_csr = handler.test_csr

# data_csr = train_csr + test_csr  # 用户真实交互稀疏矩阵

# def output_results_epoch27():
#     idx_U = list(range(0, 51))  # 前200个用户

#     trained_model = Model().cuda()
#     checkpoint = torch.load('model_save/best_model_epoch_27.pt')
#     trained_model.load_state_dict(checkpoint)

#     # 模型推理
#     U_EMDS, I_EMDS, I_EMDS_INI, I_soft_assignments, I_soft_assignments_ini = trained_model.eval_calitemclusters(adj)
#     idx_U_tensor = torch.tensor(idx_U, dtype=torch.long, device='cuda:0')
#     test_lhs = U_EMDS[idx_U_tensor]
#     test_lhs_ini, test_lhs_cluster, test_lhs_cluster_ini = trained_model.eval_caluserclusters(test_lhs, idx_U_tensor)

#     user_pair_intent = test_lhs_cluster * I_soft_assignments_ini
#     user_pair_intent = trained_model.USER_INTENT_NETWORK(user_pair_intent)

#     item_pair_intent = test_lhs_cluster_ini * I_soft_assignments
#     item_pair_intent = trained_model.ITEM_INTENT_NETWORK(item_pair_intent)

#     SOCIAL_LINK_PROB_1 = torch.cosine_similarity(test_lhs.unsqueeze(1), I_EMDS_INI, dim=2) + 1.0
#     SOCIAL_LINK_PROB_2 = torch.cosine_similarity(test_lhs_ini.unsqueeze(1), I_EMDS, dim=2) + 1.0

#     LINK_PROB = torch.cat(
#         [user_pair_intent, item_pair_intent,
#          SOCIAL_LINK_PROB_1.unsqueeze(-1), SOCIAL_LINK_PROB_2.unsqueeze(-1)],
#         dim=-1
#     )
#     LINK_PROB = trained_model.LINK_NETWORK(LINK_PROB)  # [200 x item_num x 1]
#     LINK_PROB = LINK_PROB.squeeze(-1).detach().cpu().numpy()  # [200 x item_num]

#     print(f"=== Epoch 27: 用户前20推荐命中情况 ===")
#     for i, user_id in enumerate(idx_U):
#         user_probs = LINK_PROB[i]  # 用户 i 的所有 item 的预测得分
#         top_20_idx = np.argsort(user_probs)[-20:][::-1]  # Top-20 推荐

#         true_items = data_csr.getrow(user_id).indices  # 实际交互过的 items
#         hit_count = np.intersect1d(top_20_idx, true_items).size
#         recall_ratio = hit_count / len(true_items) if len(true_items) > 0 else 0.0

#         print(f"User {user_id}: 命中 {hit_count} / {len(true_items)}，命中率 {recall_ratio:.4f}")

# output_results_epoch27()


###### 计算特定用户的hit@20的变化情况
# from evaluation import * 
# from train import Model
# handler = DataHandler()
# handler.LoadData()
# print('USER', args.user, 'ITEM', args.item)
# print('NUM OF INTERACTIONS', handler.trnLoader.dataset.__len__())
# adj = handler.torchBiAdj
# train_csr = handler.train_csr
# test_csr = handler.test_csr

# data_csr = train_csr + test_csr

# def output_user4_hits_top20():
#     user_id = 4
#     true_items = data_csr.getrow(user_id).indices  # 用户4的历史交互物品
#     hit_counts = []

#     for epoch in range(28):
#         trained_model = Model().cuda()
#         checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
#         trained_model.load_state_dict(checkpoint)

#         # 推理
#         U_EMDS, I_EMDS, I_EMDS_INI, I_soft_assignments, I_soft_assignments_ini = trained_model.eval_calitemclusters(adj)
#         user_tensor = torch.tensor([user_id], dtype=torch.long, device='cuda:0')
#         user_emb = U_EMDS[user_tensor]
#         user_emb_ini, user_cluster, user_cluster_ini = trained_model.eval_caluserclusters(user_emb, user_tensor)

#         user_pair_intent = user_cluster * I_soft_assignments_ini
#         user_pair_intent = trained_model.USER_INTENT_NETWORK(user_pair_intent)

#         item_pair_intent = user_cluster_ini * I_soft_assignments
#         item_pair_intent = trained_model.ITEM_INTENT_NETWORK(item_pair_intent)

#         sim_1 = torch.cosine_similarity(user_emb.unsqueeze(1), I_EMDS_INI, dim=2) + 1.0
#         sim_2 = torch.cosine_similarity(user_emb_ini.unsqueeze(1), I_EMDS, dim=2) + 1.0

#         link_prob = torch.cat(
#             [user_pair_intent, item_pair_intent,
#              sim_1.unsqueeze(-1), sim_2.unsqueeze(-1)],
#             dim=-1
#         )
#         link_prob = trained_model.LINK_NETWORK(link_prob)
#         link_prob = link_prob.squeeze(-1).detach().cpu().numpy().flatten()  # [item_num]

#         top_20 = np.argsort(link_prob)[-20:][::-1]
#         hit = np.intersect1d(top_20, true_items).size
#         hit_counts.append(hit)

#     # 写入结果到txt文件
#     with open("user4_hits.txt", "w") as f:
#         for epoch, hit in enumerate(hit_counts):
#             f.write(f"Epoch {epoch}: Hit@20 = {hit}\n")
#     print("写入完成，文件保存在 user4_hits.txt")

# output_user4_hits_top20()


# 计算用户4的意图分布变化
# import numpy as np
# from evaluation import * 
# from train import Model

# handler = DataHandler()
# handler.LoadData()
# print('USER', args.user, 'ITEM', args.item)
# print('NUM OF INTERACTIONS', handler.trnLoader.dataset.__len__())

# adj = handler.torchBiAdj

# def save_user4_clusters_as_matrix():
#     user_id = 4
#     cluster_list = []

#     for epoch in range(51):
#         trained_model = Model().cuda()
#         checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
#         trained_model.load_state_dict(checkpoint)

#         U_EMDS, I_EMDS, I_EMDS_INI, I_soft_assignments, I_soft_assignments_ini = trained_model.eval_calitemclusters(adj)
#         user_tensor = torch.tensor([user_id], dtype=torch.long, device='cuda:0')
#         user_emb = U_EMDS[user_tensor]
#         user_emb_ini, user_cluster, user_cluster_ini = trained_model.eval_caluserclusters(user_emb, user_tensor)

#         cluster_list.append(user_cluster.detach().cpu().numpy().flatten())

#     # 转为矩阵
#     cluster_matrix = np.vstack(cluster_list)           # 形状 (51, D)

#     # 保存为 txt 文件
#     np.savetxt("user4_test_lhs_cluster_matrix.txt", cluster_matrix, fmt="%.6f")

#     print("矩阵已保存为 txt 文件")

# save_user4_clusters_as_matrix()

##### 计算用户4在商品侧的意图变化
import numpy as np
from evaluation import * 
from train import Model

handler = DataHandler()
handler.LoadData()
print('USER', args.user, 'ITEM', args.item)
print('NUM OF INTERACTIONS', handler.trnLoader.dataset.__len__())

adj = handler.torchBiAdj

def save_user4_clusters_as_matrix():
    user_id = 4
    cluster_list = []

    for epoch in range(51):
        trained_model = Model().cuda()
        checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
        trained_model.load_state_dict(checkpoint)

        U_EMDS, I_EMDS, I_EMDS_INI, I_soft_assignments, I_soft_assignments_ini = trained_model.eval_calitemclusters(adj)
        user_tensor = torch.tensor([user_id], dtype=torch.long, device='cuda:0')
        user_emb = U_EMDS[user_tensor]
        user_emb_ini, user_cluster, user_cluster_ini = trained_model.eval_caluserclusters(user_emb, user_tensor)

        cluster_list.append(user_cluster_ini.detach().cpu().numpy().flatten())

    # 转为矩阵
    cluster_matrix = np.vstack(cluster_list)           # 形状 (51, D)

    # 保存为 txt 文件
    np.savetxt("user4_test_item_cluster_matrix.txt", cluster_matrix, fmt="%.6f")

    print("矩阵已保存为 txt 文件")

save_user4_clusters_as_matrix()



######################### 计算每个意图上的均值和方差
# import numpy as np
# import torch
# from evaluation import * 
# from train import Model

# handler = DataHandler()
# handler.LoadData()
# adj = handler.torchBiAdj
# user_num = args.user
# batch_size = 50

# def compute_and_save_cluster_stats():
#     selected_epochs = [0, 5, 27]
#     mean_matrix = []
#     var_matrix = []

#     for epoch in selected_epochs:
#         trained_model = Model().cuda()
#         checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
#         trained_model.load_state_dict(checkpoint)

#         U_EMDS, *_ = trained_model.eval_calitemclusters(adj)

#         cluster_list = []
#         for start in range(0, user_num, batch_size):
#             end = min(start + batch_size, user_num)
#             user_ids = torch.arange(start, end, dtype=torch.long, device='cuda:0')
#             user_emb = U_EMDS[user_ids]
#             _, user_cluster, _ = trained_model.eval_caluserclusters(user_emb, user_ids)
#             cluster_list.append(user_cluster.detach().cpu().numpy())

#         cluster_all = np.vstack(cluster_list)  # [user_num, D]
#         print("cluster_all shape: ", cluster_all.shape)
#         mean_vals = cluster_all.mean(axis=0)
#         var_vals = cluster_all.var(axis=0)

#         mean_matrix.append(mean_vals)
#         var_matrix.append(var_vals)

#     mean_matrix = np.vstack(mean_matrix)  # [4, D]
#     var_matrix = np.vstack(var_matrix)    # [4, D]
#     print("mean_matrix shape: ", mean_matrix.shape)
#     print("var_matrix shape: ", var_matrix.shape)

#     np.savetxt("cluster_means.txt", mean_matrix, fmt="%.6f")
#     np.savetxt("cluster_variances.txt", var_matrix, fmt="%.6f")
#     print("已保存：cluster_means.txt 和 cluster_variances.txt")

# compute_and_save_cluster_stats()

######### 计算商品侧的用户意图的平均值和方差
# import numpy as np
# import torch
# from evaluation import * 
# from train import Model

# handler = DataHandler()
# handler.LoadData()
# adj = handler.torchBiAdj
# user_num = args.user
# batch_size = 50

# def compute_and_save_cluster_stats():
#     selected_epochs = [0, 5, 27]
#     mean_matrix = []
#     var_matrix = []

#     for epoch in selected_epochs:
#         trained_model = Model().cuda()
#         checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
#         trained_model.load_state_dict(checkpoint)

#         U_EMDS, *_ = trained_model.eval_calitemclusters(adj)

#         cluster_list = []
#         for start in range(0, user_num, batch_size):
#             end = min(start + batch_size, user_num)
#             user_ids = torch.arange(start, end, dtype=torch.long, device='cuda:0')
#             user_emb = U_EMDS[user_ids]
#             _, _, user_cluster_ini = trained_model.eval_caluserclusters(user_emb, user_ids)
#             cluster_list.append(user_cluster_ini.detach().cpu().numpy())

#         cluster_all = np.vstack(cluster_list)  # [user_num, D]
#         print("cluster_all shape: ", cluster_all.shape)
#         mean_vals = cluster_all.mean(axis=0)
#         var_vals = cluster_all.var(axis=0)

#         mean_matrix.append(mean_vals)
#         var_matrix.append(var_vals)

#     mean_matrix = np.vstack(mean_matrix)  # [4, D]
#     var_matrix = np.vstack(var_matrix)    # [4, D]
#     print("mean_matrix shape: ", mean_matrix.shape)
#     print("var_matrix shape: ", var_matrix.shape)

#     np.savetxt("cluster_item_means.txt", mean_matrix, fmt="%.6f")
#     np.savetxt("cluster_item_variances.txt", var_matrix, fmt="%.6f")
#     print("已保存：cluster_item_means.txt 和 cluster_item_variances.txt")

# compute_and_save_cluster_stats()


######################### 计算每个商品上的均值和方差
# import numpy as np
# import torch
# from evaluation import * 
# from train import Model

# handler = DataHandler()
# handler.LoadData()
# adj = handler.torchBiAdj

# def compute_per_item_mean_var():
#     selected_epochs = [0, 5, 27]
#     mean_matrix = []
#     var_matrix = []
#     item_limit = 5000

#     for epoch in selected_epochs:
#         trained_model = Model().cuda()
#         checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
#         trained_model.load_state_dict(checkpoint)

#         # 获取 I_soft_assignments: [num_items, D]
#         _, _, _, _, I_soft_assignments_ini = trained_model.eval_calitemclusters(adj)

#         I_soft = I_soft_assignments_ini.squeeze(0).detach().cpu().numpy()  # 转为 numpy
#         print('I_soft shape: ', I_soft.shape)
        
#         I_soft = I_soft[:item_limit]  # 只保留前5000个商品

#         print('I_soft shape: ', I_soft.shape)

#         means = I_soft.mean(axis=1)  # 每行均值 [5000]
#         vars_ = I_soft.var(axis=1)   # 每行方差 [5000]

#         print(' means shape: ', means.shape)

#         mean_matrix.append(means)    # 累加为 [4, 5000]
#         var_matrix.append(vars_)

#     mean_matrix = np.vstack(mean_matrix)  # [4, 5000]
#     var_matrix = np.vstack(var_matrix)

#     np.savetxt("item5000_mean_per_epoch.txt", mean_matrix, fmt="%.6f")
#     np.savetxt("item5000_var_per_epoch.txt", var_matrix, fmt="%.6f")
#     print("保存成功：item5000_mean_per_epoch.txt 和 item5000_var_per_epoch.txt")

# compute_per_item_mean_var()

############ 输出商品侧的商品意图
# import numpy as np
# import torch
# from evaluation import * 
# from train import Model

# handler = DataHandler()
# handler.LoadData()
# adj = handler.torchBiAdj

# def compute_per_item_mean_var():
#     selected_epochs = [0, 5, 27]
#     mean_matrix = []
#     var_matrix = []
#     item_limit = 5000

#     for epoch in selected_epochs:
#         trained_model = Model().cuda()
#         checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
#         trained_model.load_state_dict(checkpoint)

#         # 获取 I_soft_assignments: [num_items, D]
#         _, _, _, I_soft_assignments, _ = trained_model.eval_calitemclusters(adj)

#         I_soft = I_soft_assignments.squeeze(0).detach().cpu().numpy()  # 转为 numpy
#         print('I_soft shape: ', I_soft.shape)
        
#         I_soft = I_soft[:item_limit]  # 只保留前5000个商品

#         print('I_soft shape: ', I_soft.shape)

#         means = I_soft.mean(axis=1)  # 每行均值 [5000]
#         vars_ = I_soft.var(axis=1)   # 每行方差 [5000]

#         print(' means shape: ', means.shape)

#         mean_matrix.append(means)    # 累加为 [4, 5000]
#         var_matrix.append(vars_)

#     mean_matrix = np.vstack(mean_matrix)  # [4, 5000]
#     var_matrix = np.vstack(var_matrix)

#     np.savetxt("item5000_item_mean_per_epoch.txt", mean_matrix, fmt="%.6f")
#     np.savetxt("item5000_item_var_per_epoch.txt", var_matrix, fmt="%.6f")
#     print("保存成功：item5000_item_mean_per_epoch.txt 和 item5000_item_var_per_epoch.txt")

# compute_per_item_mean_var()


#输出意图中心
# import numpy as np
# import torch
# from evaluation import * 
# from train import Model
# import os

# save_dir = 'intent_pro_outputs'
# os.makedirs(save_dir, exist_ok=True)

# epochs_to_save = [0, 5, 27]

# handler = DataHandler()
# handler.LoadData()

# for epoch in epochs_to_save:
#     trained_model = Model().cuda()
#     checkpoint = torch.load(f'model_save/best_model_epoch_{epoch}.pt')
#     trained_model.load_state_dict(checkpoint)

#     # 转为 CPU 并提取值
#     uCluster = trained_model.uCluster.detach().cpu().numpy()
#     iCluster = trained_model.iCluster.detach().cpu().numpy()

#     # 保存为 .txt 文件
#     np.savetxt(os.path.join(save_dir, f'uCluster_epoch_{epoch}.txt'), uCluster, fmt='%.6f')
#     np.savetxt(os.path.join(save_dir, f'iCluster_epoch_{epoch}.txt'), iCluster, fmt='%.6f')

#     print(f"Epoch {epoch} 的 uCluster 和 iCluster 已保存为 .txt 文件。")
