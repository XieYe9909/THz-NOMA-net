import torch
from scipy.io import savemat

gP = torch.load('data/test/gP.pt').numpy()
gS = torch.load('data/test/gS.pt').numpy()
gS = gS[:, 0] + 1j * gS[:, 1]

savemat(file_name='matlab codes/data/gP_list.mat', mdict={'gP_list': gP})
savemat(file_name='matlab codes/data/gS_list.mat', mdict={'gS_list': gS})
