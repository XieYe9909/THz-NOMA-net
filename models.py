import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch import Tensor, square, dot, log2, sum, min, relu, mul, div
from torch.optim import Adam
from torch.utils.data import DataLoader
from DataGenerator import MyDataset
from torch_geometric.nn import GATConv
from datetime import datetime


class GATNet(nn.Module):
    def __init__(self, batch_size: int,
                 N_prim: int, N_sec: int, N_antenna: int,
                 P_noise: float, P_tot: float, R_targ: float,
                 out_features_seq, heads_seq, fc_seq):
        super(GATNet, self).__init__()
        self.batch_size = batch_size
        self.N_prim = N_prim
        self.N_sec = N_sec
        self.P_noise = P_noise
        self.P_tot = P_tot
        self.R_targ = R_targ
        self.threshold = 1e-3
        self.norm_ratio = 1e6

        self.GATLayers = nn.Sequential()
        self.GATLayers.add_module(name='GAT0', module=GATConv(
            in_channels=2 * N_antenna,
            out_channels=out_features_seq[0],
            heads=heads_seq[0],
            dropout=0.2
        ))
        for i in range(len(out_features_seq) - 1):
            # self.GATLayers.add_module('ReLU{}'.format(i), nn.ReLU())
            self.GATLayers.add_module('GAT{}'.format(i + 1), GATConv(
                in_channels=out_features_seq[i] * heads_seq[i],
                out_channels=out_features_seq[i + 1],
                heads=heads_seq[i + 1],
                dropout=0.2
            ))
        # self.GATLayers.add_module(name='ReLU', module=nn.ReLU())

        self.FCLayers = nn.Sequential()
        self.FCLayers.add_module(name='FC0', module=nn.Linear(
            in_features=out_features_seq[-1] * heads_seq[-1],
            out_features=fc_seq[0]
        ))
        for j in range(len(fc_seq) - 1):
            self.FCLayers.add_module('ReLU{}'.format(j), nn.ReLU())
            self.FCLayers.add_module('Dropout{}'.format(j), nn.Dropout(p=0.2))
            self.FCLayers.add_module('FC{}'.format(j + 1), nn.Linear(
                in_features=fc_seq[j],
                out_features=fc_seq[j + 1]
            ))
        self.FCLayers.add_module(name='Dropout', module=nn.Dropout(p=0.2))
        self.FCLayers.add_module(name='ReLU', module=nn.ReLU())

    @property
    def edge_index(self):
        N = self.N_sec + 2 * self.N_prim
        set_sec = range(self.N_sec)
        set_beam = range(self.N_sec, self.N_sec + self.N_prim)
        set_prim = range(self.N_sec + self.N_prim, N)

        edge_index = torch.zeros((2, 0), dtype=torch.int)
        for i in set_sec:
            for j in set_beam:
                edge_index = torch.concat(tensors=(edge_index, torch.tensor([[i], [j]])), dim=1)
                edge_index = torch.concat(tensors=(edge_index, torch.tensor([[j], [i]])), dim=1)

        for k in range(0, len(set_prim)):
            i = set_prim[k]
            j = set_beam[k]
            edge_index = torch.concat(tensors=(edge_index, torch.tensor([[i], [j]])), dim=1)
            edge_index = torch.concat(tensors=(edge_index, torch.tensor([[j], [i]])), dim=1)

        return edge_index

    def forward(self, h, pP, pT):
        edge_index = self.edge_index
        h_GAT = torch.zeros((h.shape[0], h.shape[1], self.GATLayers[-1].out_channels * self.GATLayers[-1].heads),
                            dtype=torch.float64)
        for i in range(h.shape[0]):
            hi = h[i]
            for layer in self.GATLayers:
                hi = layer(x=hi, edge_index=edge_index)
                hi = func.relu(hi)

            h_GAT[i] = hi

        p = torch.zeros((h_GAT.shape[0], self.N_sec, self.FCLayers[-3].out_features), dtype=torch.float64)
        for n in range(self.N_sec):
            p[:, n] = self.FCLayers(h_GAT[:, n])

        ratio = div(relu(pT - sum(pP, dim=1)), sum(p, dim=(1, 2)) + 1e-5)
        ratio = torch.where(torch.gt(ratio, other=1), 1, ratio)
        p = mul(p, ratio.view((-1, 1, 1)))

        p = torch.where(torch.gt(p, self.threshold), p, 0)
        return p

    def loss_func(self, pP: Tensor, pS: Tensor, pN: float, gP: Tensor, gS: Tensor):
        K = self.N_prim
        M = self.N_sec

        gS2 = square(gS[:, 0]) + square(gS[:, 1])
        gP2 = square(gP)

        cross_prim = torch.zeros((self.batch_size, M))
        cross_sec = torch.zeros((self.batch_size, M, M))
        for b in range(0, self.batch_size):
            for m in range(0, M):
                cross_prim[b, m] = dot(gS2[b, m], pP[b])
                for j in range(0, M):
                    cross_sec[b, m, j] = square(dot(gS[b, 0, m], torch.sqrt(pS[b, j]))) + \
                                         square(dot(gS[b, 1, m], torch.sqrt(pS[b, j])))

        rate_sec = torch.zeros((self.batch_size, M))
        for b in range(0, self.batch_size):
            for m in range(0, M):
                rate_sec[b, m] = log2(1 + cross_sec[b, m, m] / (cross_prim[b, m] +
                                                                sum(cross_sec[b, m]) - cross_sec[b, m, m] + pN))

        pS = torch.sort(pS, dim=1, descending=True)[0]
        rate_p2s = 100 * torch.ones((self.batch_size, M, K))  # 100是虛擬上限
        for b in range(0, self.batch_size):
            for k in range(0, K):
                m = 0
                while m < M and pS[b, m, k] > 0:
                    rate_p2s[b, m, k] = log2(1 + pS[b, m, k] / (pP[b, k] + sum(pS[b, m + 1: M, k]) + pN / gP2[b, k]))
                    m += 1

        rate_final = torch.zeros_like(rate_sec)
        for b in range(self.batch_size):
            for m in range(0, M):
                rate_final[b, m] = min(torch.cat((rate_sec[b, m].unsqueeze(dim=0), rate_p2s[b, m])))

        rate_tot = sum(rate_final, dim=1)
        loss = -rate_tot
        return rate_tot, torch.mean(loss)

    def net_train(self, dataset: MyDataset, learning_rate=1e-4, total_epochs=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        print(f"Using {device.type} device.")

        train_batch = DataLoader(dataset=dataset, batch_size=self.batch_size)
        batch_num = len(train_batch)

        pN = self.P_noise * self.norm_ratio ** 2
        pT = self.P_tot
        R = self.R_targ

        loss_log = np.zeros((total_epochs,))
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for epoch in range(0, total_epochs):
            loss_epoch = 0
            for h, gP, gS in train_batch:
                h[:, :self.N_sec] *= self.norm_ratio
                h[:, self.N_sec: self.N_sec + self.N_prim] *= 100
                h[:, -self.N_prim:] *= self.norm_ratio

                gP *= self.norm_ratio
                gS *= self.norm_ratio

                gP = gP.to(device)
                gS = gS.to(device)

                optimizer.zero_grad()
                pP = (2 ** R - 1) * pN / square(gP)
                pS = self.forward(h, pP=pP, pT=pT)
                loss = self.loss_func(pP=pP, pS=pS, pN=pN, gP=gP, gS=gS)[1]
                loss_epoch += loss.item()

                loss.backward()
                optimizer.step()

            loss_log[epoch] = loss_epoch / batch_num
            print('Epochs:', epoch + 1, 'Loss =', loss_log[epoch])

        now = datetime.now()
        date_time = now.strftime('%m-%d-%H%M')
        torch.save(self.state_dict(), f='saved nets/gnn_' + date_time + '.pth')
        np.save(file='train logs/loss_log_' + date_time + '.npy', arr=loss_log)

    def net_test(self, date_time: str, dataset: MyDataset):
        self.load_state_dict(torch.load('saved nets/gnn_' + date_time + '.pth'))

        pN = self.P_noise * self.norm_ratio ** 2
        pT = self.P_tot
        R = self.R_targ

        test_data = DataLoader(dataset=dataset, batch_size=1)
        test_result = []

        for h, gP, gS in test_data:
            h[:, :self.N_sec] *= self.norm_ratio
            h[:, -self.N_prim:] *= self.norm_ratio

            gP *= self.norm_ratio
            gS *= self.norm_ratio

            pP = (2 ** R - 1) * pN / square(gP)
            pS = self.forward(h, pP=pP, pT=pT)
            tot_rate = self.loss_func(pP=pP, pS=pS, pN=pN, gP=gP, gS=gS)[0]
            test_result.append(tot_rate[0].item())

        np.save(file='test logs/result_' + date_time + '.npy', arr=test_result)
        return np.mean(test_result)
