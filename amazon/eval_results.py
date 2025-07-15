from evaluation import *
from train import Model


handler = DataHandler()
handler.LoadData()
print('USER', args.user, 'ITEM', args.item)
print('NUM OF INTERACTIONS', handler.trnLoader.dataset.__len__())
adj = handler.torchBiAdj
train_csr = handler.train_csr
test_csr = handler.test_csr

print_max_K = 0

for epoch in range(0, 30):
    trained_model = Model().cuda()
    checkpoint = torch.load('model_save/best_model_epoch_'+str(epoch)+'.pt')
    trained_model.load_state_dict(checkpoint)

    best_ndcg = 0.
    best_metrics = {}
    MAX_K = max(args.topk)
    target_metric = f'ndcg@{MAX_K}'

    metrics = batch_evaluation(trained_model, test_csr, train_csr, adj, epoch, args.tstBat, args.topk, MAX_K)

    if metrics[target_metric] >= best_ndcg:
        best_metrics = metrics.copy()
        best_ndcg = metrics[target_metric]

    print('Epoch', epoch, '|', end='\t')
    print_metrics(metrics, args.topk, MAX_K, print_max_K)
    print('** best performance: epoch', best_metrics['epoch'], '**')
    print('Epoch', best_metrics['epoch'], '|', end='\t')
    print_metrics(best_metrics, args.topk, MAX_K, print_max_K)
