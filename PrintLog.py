import numpy as np
import matplotlib.pyplot as plt

loss = np.load('train logs/loss_log_03-21-1530.npy')

plt.plot(np.arange(loss.shape[0]), loss)
plt.xlabel('Epochs', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.show()
