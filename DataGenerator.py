import numpy as np
import torch
from torch.utils.data import Dataset
from TransmitModel import UserState, ChannelModel


class MyDataset(Dataset):
    def __init__(self, data_type='train'):
        assert data_type in ['train', 'test']
        data_path = 'data/' + data_type
        self.h_list = torch.load(data_path + '/h.pt')
        self.gP_list = torch.load(data_path + '/gP.pt')
        self.gS_list = torch.load(data_path + '/gS.pt')
        self.N_antenna = self.h_list.shape[-1] // 2
        self.N_sec = self.gS_list.shape[2]
        self.N_prim = self.gS_list.shape[3]

    def __getitem__(self, index: int):
        h = self.h_list[index]
        gP = self.gP_list[index]
        gS = self.gS_list[index]
        return h, gP, gS

    def __len__(self):
        return self.gP_list.shape[0]


def generate_data(cm: ChannelModel, data_num: int, data_type='train'):
    assert data_type in ['train', 'test']
    K = cm.num_prim
    M = cm.num_sec
    N = cm.num_antenna

    h = np.zeros(shape=(data_num, M + 2 * K, 2 * N))
    gP = np.zeros(shape=(data_num, K))
    gS = np.zeros(shape=(data_num, 2, M, K))
    for n in range(0, data_num):
        cm.state_prim = UserState(cm.num_prim, cm.range_prim, random_angle=False, random_fading=False)
        cm.state_sec = UserState(cm.num_sec, cm.range_sec, random_angle=True, random_fading=False)

        h[n, :M, :N] = np.real(cm.channel_sec)
        h[n, :M, N:] = np.imag(cm.channel_sec)
        h[n, M: M + K, :N] = np.real(cm.beams)
        h[n, M: M + K, N:] = np.imag(cm.beams)
        h[n, M + K:, :N] = np.real(cm.channel_prim)
        h[n, M + K:, N:] = np.imag(cm.channel_prim)

        gP[n] = cm.gain_prim
        gS[n, 0] = np.real(cm.gain_sec)
        gS[n, 1] = np.imag(cm.gain_sec)

    h = torch.tensor(h)
    gP = torch.tensor(gP)
    gS = torch.tensor(gS)

    path = 'data/' + data_type
    torch.save(h, f=path + '/h.pt')
    torch.save(gP, f=path + '/gP.pt')
    torch.save(gS, f=path + '/gS.pt')


if __name__ == '__main__':
    channel_model = ChannelModel(N=64, K=19, M=10, NQ=64, RP=20, RS=20)
    # generate_data(channel_model, data_num=5000, data_type='train')
    generate_data(channel_model, data_num=100, data_type='test')
