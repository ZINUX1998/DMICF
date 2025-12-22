from evaluation import *
from train import Model

def output_results():
    handler = DataHandler()
    handler.LoadData()
    print('USER', args.user, 'ITEM', args.item)
    print('NUM OF INTERACTIONS', handler.trnLoader.dataset.__len__())
    adj = handler.torchBiAdj
    uu_adj = handler.uu_csr_tensor
    ii_adj = handler.ii_csr_tensor
    train_csr = handler.train_csr
    test_csr = handler.test_csr
    
    epoch = 13
    trained_model = Model(adj, uu_adj, ii_adj).cuda()
    checkpoint = torch.load('model_save/best_model_epoch_'+str(epoch)+'.pt')
    trained_model.load_state_dict(checkpoint)
    
    u_sim_embs, i_agg_embs, u_agg_embs, i_sim_embs, u_sim_intents, i_agg_intents, u_agg_intents, i_sim_intents = trained_model.eval_calclusteres()

    return u_sim_intents, i_agg_intents, u_agg_intents, i_sim_intents

u_sim, i_agg, u_agg, i_sim = output_results()
np.save("ml10m_u_sim.npy", u_sim.cpu().numpy())
np.save("ml10m_i_agg.npy", i_agg.cpu().numpy())
np.save("ml10m_u_agg.npy", u_agg.cpu().numpy())
np.save("ml10m_i_sim.npy", i_sim.cpu().numpy())