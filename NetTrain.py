from models import GATNet
from DataGenerator import MyDataset


train_data = MyDataset(data_type='train')
N = train_data.N_antenna
M = train_data.N_sec
K = train_data.N_prim

gnn = GATNet(batch_size=5, N_prim=K, N_sec=M, N_antenna=N, P_noise=1e-12, P_tot=1, R_targ=1,
             out_features_seq=[100, 150, 200],
             heads_seq=[2, 2, 2],
             fc_seq=[300, 100, K]
             ).double()
gnn.net_train(train_data, learning_rate=1e-4, total_epochs=100)
