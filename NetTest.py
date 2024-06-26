from models import GATNet
from DataGenerator import MyDataset
from datetime import datetime

test_data = MyDataset(data_type='test')
N = test_data.N_antenna
M = test_data.N_sec
K = test_data.N_prim

start = datetime.now()
gnn = GATNet(batch_size=1, N_prim=K, N_sec=M, N_antenna=N, P_noise=1e-12, P_tot=1, R_targ=1,
             out_features_seq=[100, 150, 200],
             heads_seq=[2, 2, 2],
             fc_seq=[300, 100, K]
             ).double()
result = gnn.net_test(date_time='04-18-2328', dataset=test_data)
end = datetime.now()
diff = end - start
print(diff)
print(result)
