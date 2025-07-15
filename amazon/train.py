from evaluation import *
set_random_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0'

torch.cuda.empty_cache()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        init = nn.init.xavier_uniform_
        
        self.uEmbeds_ini = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds_ini = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.uCluster = nn.Parameter(init(t.empty(args.intentNum, args.latdim)))
        self.iCluster = nn.Parameter(init(t.empty(args.intentNum, args.latdim)))
        
        CLUSTER_architecture = [args.intentNum, args.MiddleIntentnum, args.FinedintentNum]
        self.U_CLUSTER_NETWORK_USER = MLP(CLUSTER_architecture)
        self.I_CLUSTER_NETWORK_ITEM = MLP(CLUSTER_architecture)
        self.U_CLUSTER_NETWORK_ITEM = MLP(CLUSTER_architecture)
        self.I_CLUSTER_NETWORK_USER = MLP(CLUSTER_architecture)
        
        INTENT_architecture = [args.FinedintentNum, 128, 64, args.intent_dim]
        self.USER_INTENT_NETWORK = MLP(INTENT_architecture)
        self.ITEM_INTENT_NETWORK = MLP(INTENT_architecture)
        self.LINK_NETWORK = MLP([2*args.intent_dim + 2, 32, 1])
        
    def calclusteres(self, user_embs, user_embs_ini, item_embs, item_embs_ini):
        
        U_soft_assignments = F.cosine_similarity(user_embs.unsqueeze(1), self.uCluster.unsqueeze(0), dim=2)
        U_soft_assignments = self.U_CLUSTER_NETWORK_USER(U_soft_assignments)
        U_soft_assignments = (U_soft_assignments.t() / torch.sum(U_soft_assignments, 1)).t()

        U_soft_assignments_ini = F.cosine_similarity(user_embs_ini.unsqueeze(1), self.iCluster.unsqueeze(0), dim=2)
        U_soft_assignments_ini = self.I_CLUSTER_NETWORK_USER(U_soft_assignments_ini)
        U_soft_assignments_ini = (U_soft_assignments_ini.t() / torch.sum(U_soft_assignments_ini, 1)).t()

        I_soft_assignments = F.normalize(item_embs, p=2, dim=-1)
        cluster_norm = F.normalize(self.iCluster, p=2, dim=-1)
        I_soft_assignments = torch.matmul(I_soft_assignments, cluster_norm.T)
        I_soft_assignments = self.I_CLUSTER_NETWORK_ITEM(I_soft_assignments) # (batch_size, 1 + neg_num, cluster_numbers)
        #sum_batch = torch.sum(I_soft_assignments, dim=2, keepdim=True)
        #I_soft_assignments = I_soft_assignments / sum_batch

        I_soft_assignments_ini = F.normalize(item_embs_ini, p=2, dim=-1)
        cluster_norm = F.normalize(self.uCluster, p=2, dim=-1)
        I_soft_assignments_ini = torch.matmul(I_soft_assignments_ini, cluster_norm.T)
        I_soft_assignments_ini = self.U_CLUSTER_NETWORK_ITEM(I_soft_assignments_ini) # (batch_size, 1 + neg_num, cluster_numbers)
        #sum_batch = torch.sum(I_soft_assignments_ini, dim=2, keepdim=True)
        #I_soft_assignments_ini = I_soft_assignments_ini / sum_batch
        
        return U_soft_assignments, U_soft_assignments_ini, I_soft_assignments, I_soft_assignments_ini
    
    def eval_calitemclusters(self, adj):
        node_embeds = t.concat([self.uEmbeds_ini, self.iEmbeds_ini], dim=0)
        embeds = (t.spmm(adj, node_embeds))
        uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]

        I_soft_assignments = F.cosine_similarity(iEmbeds.unsqueeze(1), self.iCluster.unsqueeze(0), dim=2)
        I_soft_assignments = self.I_CLUSTER_NETWORK_ITEM(I_soft_assignments)
        #I_soft_assignments = (I_soft_assignments.t() / torch.sum(I_soft_assignments, 1)).t()
        
        I_soft_assignments_ini = F.cosine_similarity(self.iEmbeds_ini.unsqueeze(1), self.uCluster.unsqueeze(0), dim=2)
        I_soft_assignments_ini = self.U_CLUSTER_NETWORK_ITEM(I_soft_assignments_ini)
        #I_soft_assignments_ini = (I_soft_assignments_ini.t() / torch.sum(I_soft_assignments_ini, 1)).t()

        return uEmbeds.detach(), iEmbeds.detach().unsqueeze(0), self.iEmbeds_ini.detach().unsqueeze(0), I_soft_assignments.detach().unsqueeze(0), I_soft_assignments_ini.detach().unsqueeze(0)

    def eval_caluserclusters(self, user_embs, user_index):
        U_soft_assignments = F.cosine_similarity(user_embs.unsqueeze(1), self.uCluster.unsqueeze(0), dim=2)
        U_soft_assignments = self.U_CLUSTER_NETWORK_USER(U_soft_assignments)
        U_soft_assignments = (U_soft_assignments.t() / torch.sum(U_soft_assignments, 1)).t()

        batch_user_embeds = self.uEmbeds_ini[user_index].detach()
        U_soft_assignments_ini = F.cosine_similarity(batch_user_embeds.unsqueeze(1), self.iCluster.unsqueeze(0), dim=2)
        U_soft_assignments_ini = self.I_CLUSTER_NETWORK_USER(U_soft_assignments_ini)
        U_soft_assignments_ini = (U_soft_assignments_ini.t() / torch.sum(U_soft_assignments_ini, 1)).t()

        return batch_user_embeds, U_soft_assignments.detach().unsqueeze(1), U_soft_assignments_ini.detach().unsqueeze(1)


    def calcLosses(self, head_index, tail_index, pos_tail_index, adj): 
        # 节点自身的嵌入
        node_embeds = t.concat([self.uEmbeds_ini, self.iEmbeds_ini], dim=0)
        embeds = (t.spmm(adj, node_embeds))
        self.uEmbeds, self.iEmbeds = embeds[:args.user], embeds[args.user:]
        
        lhs = self.uEmbeds[head_index]     # (batch_size, embedding_dim)
        rhs = self.iEmbeds[tail_index]     # (batch_size, neg_num+1, embedding_dim)
        lhs_ini = self.uEmbeds_ini[head_index]
        rhs_ini = self.iEmbeds_ini[tail_index]
        
        lhs_clusters, lhs_clusters_ini, rhs_clusters, rhs_clusters_ini = self.calclusteres(lhs, lhs_ini, rhs, rhs_ini) 
        
        ############ 计算重构损失

        user_pair_intents = lhs_clusters.unsqueeze(1) * rhs_clusters_ini
        user_pair_intents = self.USER_INTENT_NETWORK(user_pair_intents)

        item_pair_intents = lhs_clusters_ini.unsqueeze(1) * rhs_clusters
        item_pair_intents = self.ITEM_INTENT_NETWORK(item_pair_intents)

        social_link_prob_1 = F.cosine_similarity(lhs.unsqueeze(1), rhs_ini, dim=2) + 1
        social_link_prob_2 = F.cosine_similarity(lhs_ini.unsqueeze(1), rhs, dim=2) + 1

        LINK_PROB = torch.cat([user_pair_intents, item_pair_intents, social_link_prob_1.unsqueeze(-1), social_link_prob_2.unsqueeze(-1)], dim=-1)
        LINK_PROB = self.LINK_NETWORK(LINK_PROB)
        LINK_PROB = LINK_PROB.squeeze(-1)
        LINK_PROB = torch.exp(LINK_PROB / args.temp)
        LINK_PROB = (LINK_PROB.t() / torch.sum(LINK_PROB, 1)).t()

        return LINK_PROB.view(-1)
        

class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())

        self.adj = self.handler.torchBiAdj
        self.train_csr = self.handler.train_csr
        self.test_csr = self.handler.test_csr

        torch.save(self.adj, 'model_save/torch_bi_adj.pt')  # 保存生成的 torchBiAdj
        np.save('model_save/train_csr.npy', self.train_csr)  # 保存训练集csr矩阵
        np.save('model_save/test_csr.npy', self.test_csr)    # 保存测试集csr矩阵
        
        self.model = Model().cuda()
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_function = torch.nn.MSELoss(reduction='sum')

    def run(self):
        trnLoader = self.handler.trnLoader
        self.model.train()
        
        for epoch in range(args.epoch):
            total_loss = 0.
        
            for batch_id, batch in enumerate(trnLoader):
                idx_U, pos_idx_V, neg_idx_V = batch
                V_idx = torch.cat((pos_idx_V.unsqueeze(dim=1), neg_idx_V), dim=1).to(device)
                pos_lables = torch.ones_like(pos_idx_V).to(device)
                neg_lables = torch.zeros_like(neg_idx_V).to(device)
                true_labels = torch.cat((pos_lables.unsqueeze(dim=1), neg_lables), dim=1).to(device)
                true_labels = true_labels.float().to(device)
                
                pairSimilarity = self.model.calcLosses(idx_U, V_idx, pos_idx_V, self.adj)
                loss = self.loss_function(pairSimilarity.to(device), true_labels.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if batch_id % 100 == 0:
                    print('############ batch : ', batch_id)
                    print('pair loss : ', loss.item())
                
            print(f'Epoch: {epoch:02d}, Loss: {total_loss:.4f}')

            model_path = f"model_save/best_model_epoch_{epoch}.pt"
            torch.save(self.model.state_dict(), model_path)


if __name__ == '__main__':
    handler = DataHandler()
    handler.LoadData()
    coach = Coach(handler)
    best_prediction = coach.run()
