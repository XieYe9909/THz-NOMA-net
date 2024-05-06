# Instructions
## Functions of main python modules
- ***TransmitModel.py***  
  Generate a channel realization of given parameters like $K, M, N$, which are the number of primary users (beams), secondary users and antennas, respectively.
  
- ***DataGenerator.py***  
  Generate train/test datasets of specified size. A dataset is expressed as a tensor $h$ in shape $(T, M+2K, 2N)$, where $T$ is the size of the dataset (or the number of different channel realizations). For each entry of $h$, it is a concatenation of $h^S_{1:M}, w_{1:K}$ and $h^P_{1:K}$, which are the CSI of all secondary users, all beamforming vectors and the CSI of all primary users.

- ***models.py***  
  Define the network architecture, the process of forward propogation, the loss function (based on the objective function of our particular problem), as well as the process of network training and performance test of a trained network.
  
- ***NetTrain.py*** & ***NetTest.py***  
  Import data from saved dataset for network training or test a trained network.

## GNN architecture
The network architecture is set in the part *GATNet* in *models.py*. Note that our network is the combination of multiple number of GAT layers and multiple number of fully connected (FC) layers. Denote by $N_{GAT}$ and $N_{FC}$ the number of GAT layers and FC layers.

The main paremeters are *out_features_seq*, *heads_seq* and *fc_seq*. *out_features_seq* and *heads_seq* are two arrays of length $N_{GAT}$, which define the number of output features and number of heads of each GAT layer, respectively. *fc_seq* is an array of length $N_{FC}$, which defines the output size of each FC layer.

> Remark: The number of input features of the $i$-th GAT layer is *out_features_seq[i-1] \* heads_seq[i-1]* ($2N$ for the first GAT layer). Also, the input size of the $j$-th FC layer is *fc_seq[j-1]* (*out_features_seq*[ $N_{GAT}$ ] \* *heads_seq*[ $N_{GAT}$ ] for the first FC layer).
