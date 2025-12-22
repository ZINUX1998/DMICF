from evaluation import *
set_random_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0'

torch.cuda.empty_cache()


class FineIntentEncoder(nn.Module):
    def __init__(self, in_dim, middle_dim, out_dim):
        super().__init__()
        self.mu_mlp = MLP([in_dim, middle_dim, out_dim])
        self.logvar_mlp = MLP([in_dim, middle_dim, out_dim])

    def forward(self, x, training=True):
        mu = self.mu_mlp(x)
        logvar = self.logvar_mlp(x)
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # reparameterization trick
            return z, mu, logvar
        else:
            z = mu
            return z


class Model(nn.Module):
    def __init__(self, adj, user_sim_adj, item_sim_adj):
        super(Model, self).__init__()

        self.adj = adj
        self.user_sim_adj = user_sim_adj
        self.item_sim_adj = item_sim_adj

        init = nn.init.xavier_uniform_
        
        self.uEmbeds_ini = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds_ini = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.uCluster = nn.Parameter(init(t.empty(args.intentNum, args.latdim)))
        self.iCluster = nn.Parameter(init(t.empty(args.intentNum, args.latdim)))
        
        self.user_sim_intent_encoder = FineIntentEncoder(args.intentNum, args.MiddleIntentnum, args.FinedintentNum)
        self.item_agg_intent_encoder = FineIntentEncoder(args.intentNum, args.MiddleIntentnum, args.FinedintentNum)
        self.item_sim_intent_encoder = FineIntentEncoder(args.intentNum, args.MiddleIntentnum, args.FinedintentNum)
        self.user_agg_intent_encoder = FineIntentEncoder(args.intentNum, args.MiddleIntentnum, args.FinedintentNum)

        
        intent_alignment_architecture = [args.FinedintentNum, 128, 64, args.intent_dim]
        self.user_intent_alignment = MLP(intent_alignment_architecture)
        self.item_intent_alignment = MLP(intent_alignment_architecture)

        self.intent_fusion_mlp = MLP([2*args.intent_dim + 2, 32, 1])
        
    def eval_calclusteres(self):
        node_embeds = t.concat([self.uEmbeds_ini, self.iEmbeds_ini], dim=0)
        embeds = (t.spmm(self.adj, node_embeds))
        user_agg_embs, item_agg_embs = embeds[:args.user], embeds[args.user:]
        user_sim_embs = (t.spmm(self.user_sim_adj, self.uEmbeds_ini))
        item_sim_embs = (t.spmm(self.item_sim_adj, self.iEmbeds_ini))

        uAnchor_norm = F.normalize(self.uCluster, p=2, dim=1) # [A, D]
        iAnchor_norm = F.normalize(self.iCluster, p=2, dim=1) # [A, D]
        
        user_sim_intent = F.normalize(user_sim_embs, p=2, dim=1)   # 用户聚合用户
        item_sim_intent = F.normalize(item_sim_embs, p=2, dim=1)   # 商品聚合商品
        user_agg_intent = F.normalize(user_agg_embs, p=2, dim=1)  # 用户聚合商品
        item_agg_intent = F.normalize(item_agg_embs, p=2, dim=1)  # 商品聚合用户

        user_sim_intent = torch.matmul(user_sim_intent, uAnchor_norm.T)
        item_agg_intent = torch.matmul(item_agg_intent, uAnchor_norm.T) 
        item_sim_intent = torch.matmul(item_sim_intent, iAnchor_norm.T)
        user_agg_intent = torch.matmul(user_agg_intent, iAnchor_norm.T)

        user_sim_intent = self.user_sim_intent_encoder(user_sim_intent, False)
        item_agg_intent = self.item_agg_intent_encoder(item_agg_intent, False)
        item_sim_intent = self.item_sim_intent_encoder(item_sim_intent, False)
        user_agg_intent = self.user_agg_intent_encoder(user_agg_intent, False)

        return user_sim_embs.detach(), item_agg_embs.detach(), user_agg_embs.detach(), item_sim_embs.detach(), user_sim_intent.detach(), item_agg_intent.detach(), user_agg_intent.detach(), item_sim_intent.detach()
        
    def contrastive_loss(self, z1, z2, temp=0.1):
        # z1, z2: [B, D]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = torch.matmul(z1, z2.T) / temp   # [B, B]
        labels = torch.arange(z1.size(0)).long().to(z1.device)
        return F.cross_entropy(logits, labels)
    
    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def calcLosses(self, head_index, tail_index): 
        # 节点自身的嵌入
        node_embeds = t.concat([self.uEmbeds_ini, self.iEmbeds_ini], dim=0)
        embeds = (t.spmm(self.adj, node_embeds))
        user_agg_embs, item_agg_embs = embeds[:args.user], embeds[args.user:]
        user_sim_embs = (t.spmm(self.user_sim_adj, self.uEmbeds_ini))
        item_sim_embs = (t.spmm(self.item_sim_adj, self.iEmbeds_ini))

        uAnchor_norm = F.normalize(self.uCluster, p=2, dim=1) # [A, D]
        iAnchor_norm = F.normalize(self.iCluster, p=2, dim=1) # [A, D]
        
        user_sim_intent = F.normalize(user_sim_embs, p=2, dim=1)   # 用户聚合用户
        item_sim_intent = F.normalize(item_sim_embs, p=2, dim=1)   # 商品聚合商品
        user_agg_intent = F.normalize(user_agg_embs, p=2, dim=1)  # 用户聚合商品
        item_agg_intent = F.normalize(item_agg_embs, p=2, dim=1)  # 商品聚合用户

        user_sim_intent = torch.matmul(user_sim_intent, uAnchor_norm.T)
        item_agg_intent = torch.matmul(item_agg_intent, uAnchor_norm.T) 
        item_sim_intent = torch.matmul(item_sim_intent, iAnchor_norm.T)
        user_agg_intent = torch.matmul(user_agg_intent, iAnchor_norm.T)

        user_sim_intent, mu_uu, logvar_uu = self.user_sim_intent_encoder(user_sim_intent)
        item_agg_intent, mu_iu, logvar_iu = self.item_agg_intent_encoder(item_agg_intent)
        item_sim_intent, mu_ii, logvar_ii = self.item_sim_intent_encoder(item_sim_intent)
        user_agg_intent, mu_ui, logvar_ui = self.user_agg_intent_encoder(user_agg_intent)

        user_sim_intent = user_sim_intent[head_index]
        item_agg_intent = item_agg_intent[tail_index]
        user_agg_intent = user_agg_intent[head_index]
        item_sim_intent = item_sim_intent[tail_index]

        batch_user_sim_embs = user_sim_embs[head_index]
        batch_item_sim_embs = item_sim_embs[tail_index]
        batch_user_agg_embs = user_agg_embs[head_index]
        batch_item_agg_embs = item_agg_embs[tail_index]

        user_pair_intents = user_sim_intent.unsqueeze(1) * item_agg_intent
        user_pair_intents = self.user_intent_alignment(user_pair_intents)
        item_pair_intents = user_agg_intent.unsqueeze(1) * item_sim_intent
        item_pair_intents = self.item_intent_alignment(item_pair_intents)

        user_structure_sim = F.cosine_similarity(batch_user_sim_embs.unsqueeze(1), batch_item_agg_embs, dim=2) + 1
        item_structure_sim = F.cosine_similarity(batch_user_agg_embs.unsqueeze(1), batch_item_sim_embs, dim=2) + 1
    
        intent_fusion_feature = torch.cat([user_pair_intents, item_pair_intents, user_structure_sim.unsqueeze(-1), item_structure_sim.unsqueeze(-1)], dim=-1)
        intent_fusion_feature = self.intent_fusion_mlp(intent_fusion_feature)
        intent_fusion_feature = intent_fusion_feature.squeeze(-1)
        intent_fusion_feature = torch.exp(intent_fusion_feature / args.temp)
        intent_fusion_feature = (intent_fusion_feature.t() / torch.sum(intent_fusion_feature, 1)).t()

        kl_loss_total = (
            self.kl_loss(mu_uu[head_index], logvar_uu[head_index]) +
            self.kl_loss(mu_iu[tail_index[:,0]], logvar_iu[tail_index[:,0]]) +
            self.kl_loss(mu_ii[tail_index[:,0]], logvar_ii[tail_index[:,0]]) +
            self.kl_loss(mu_ui[head_index], logvar_ui[head_index])
        )

        return intent_fusion_feature.view(-1), kl_loss_total
        

class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())

        adj = self.handler.torchBiAdj
        uu_adj = self.handler.uu_csr_tensor
        ii_adj = self.handler.ii_csr_tensor
        self.train_csr = self.handler.train_csr
        self.test_csr = self.handler.test_csr

        # torch.save(self.adj, 'model_save/torch_bi_adj.pt')  # 保存生成的 torchBiAdj
        # np.save('model_save/train_csr.npy', self.train_csr)  # 保存训练集csr矩阵
        # np.save('model_save/test_csr.npy', self.test_csr)    # 保存测试集csr矩阵
        
        self.model = Model(adj, uu_adj, ii_adj).cuda()
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_function = torch.nn.MSELoss(reduction='sum')

    def run(self, print_max_K = 0):
        best_ndcg = 0.
        best_metrics = {}
        MAX_K = max(args.topk)
        target_metric = f'ndcg@{MAX_K}'
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
                
                # pairSimilarity, contrastive_loss = self.model.calcLosses(idx_U, V_idx)
                pairSimilarity, kl_loss = self.model.calcLosses(idx_U, V_idx)
                pair_loss = self.loss_function(pairSimilarity.to(device), true_labels.view(-1))
                loss = pair_loss + 100 * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if batch_id % 100 == 0:
                    print('############ batch : ', batch_id)
                    print('pair loss : ', pair_loss.item(), "    kl loss: ", kl_loss.item())
                    # print('pair loss : ', loss.item())
                
            print(f'Epoch: {epoch:02d}, Loss: {total_loss:.4f}')

            model_path = f"model_save/best_model_epoch_{epoch}.pt"
            torch.save(self.model.state_dict(), model_path)
            
            metrics = batch_evaluation(self.model, self.test_csr, self.train_csr, epoch, args.tstBat, args.topk, MAX_K)
            
            if metrics[target_metric] >= best_ndcg:
                best_metrics = metrics.copy()
                best_ndcg = metrics[target_metric]

            print('** epoch', epoch, 'total_loss: ', total_loss, '**')
            print('Epoch', epoch, '|', end='\t')
            print_metrics(metrics, args.topk, MAX_K, print_max_K)
            print('** best performance: epoch', best_metrics['epoch'], '**')
            print('Epoch', best_metrics['epoch'], '|', end='\t')
            print_metrics(best_metrics, args.topk, MAX_K, print_max_K)

if __name__ == '__main__':
    handler = DataHandler()
    handler.LoadData()
    coach = Coach(handler)
    best_prediction = coach.run()
