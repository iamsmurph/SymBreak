import torch
from torch.nn.modules import dropout
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import torch.nn as nn
import random
import torchvision.transforms.functional as TF
import math

dropout_prob = 0.25

class DiscriminatorDataset(Dataset):

    def __init__(self, transform=None, train=None):
        df = np.load("datasets/dfRandomVer1Rot90Dipole.npy")
        
        X = df[:, :-1].astype(np.float32)
        binaryTargs = (df[:, -1] > np.median(df[:, -1]))*1
        y = binaryTargs.astype(np.float32).reshape(-1, 1)
        # CHANGE TO MEDIAN LATER FOR ADDITIONAL TESTING

        self.n_samples = X.shape[0]
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ComplementLayer(nn.Module):
    """ Custom layer for ensuring complementary probabilities """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        self.weights = torch.Tensor(size_out, size_in)
        
        self.bias = torch.Tensor(size_out)
        
        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        weights_flip = torch.flip(self.weights, [0,1])
        weights_cat = torch.cat((self.weights, weights_flip))

        bias_flip = torch.flip(self.bias, [0])
        bias_cat = torch.cat((self.bias, bias_flip))

        w_times_x = torch.mm(x, weights_cat.t())
        return torch.add(w_times_x, bias_cat)

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        
        # for feature extraction model
        self.l1 = nn.Linear(input_size, 20, bias=True)
        self.l2 = nn.Linear(20, 4, bias=True)

        # stitching layer
        self.l_stitch = ComplementLayer(8, 2)
    
    def forward(self, x1, x2):
        out1 = self.l1(x1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        out1 = self.l2(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)

        out2 = self.l1(x2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        out2 = self.l2(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)

        out2_flip = torch.flip(out2, [0])
        feats = torch.cat((out1, out2_flip))
        
        out_final = self.l_stitch(feats)
        return out_final

inputVec = np.random.rand((100,3))
targetVec = np.round(np.random.rand(100), 0)


