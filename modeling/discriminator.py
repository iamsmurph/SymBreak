import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import math

# Hyper-parameters 
hparam = {"input_size": 3, "output_size": 1, "num_epochs": 1000, 
            "batch_size": 60, "lr": 1e-4, "dropout_prob": 0.25, "lrDecay": 0.995, 
            "weight_decay": 1e-6, "valProp": .2, "trShuffle": True, "tstShuffle": False}

class DiscriminatorDataset(Dataset):

    def __init__(self, transform=None, train=None):
        df = np.load("datasets/dfRandomVer1Rot90Dipole.npy")
        
        X = df[:, 2:-2].astype(np.float32)
        TargClasses = df[:, -2:]
        y = TargClasses.astype(np.float32).reshape(-1, 2)

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
        assert(size_in % 2 == 0)
        assert(size_out % 2 == 0)
        
        self.size_in, self.size_out = size_in, size_out

        self.weights = torch.Tensor(size_out, size_in // 2)
        
        self.bias = torch.Tensor(size_out // 2)
        
        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  

    def forward(self, x):
        weights_flip = torch.flip(self.weights, [0,1])
        weights_cat = torch.cat((self.weights, weights_flip), 1)

        bias_flip = torch.flip(self.bias, [0])
        bias_cat = torch.cat((self.bias.reshape(-1,1), bias_flip.reshape(-1,1)))

        w_times_x = torch.mm(x, weights_cat.t())
        return torch.add(w_times_x, bias_cat)

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hparam["dropout_prob"])
        
        # for feature extraction model
        self.l1 = nn.Linear(input_size, 10, bias=True)
        self.l2 = nn.Linear(10, 4, bias=True)

        # final layers
        self.lf1 = ComplementLayer(8, 10)
        self.lf2 = ComplementLayer(10, 2)
    
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
        
        out_final = self.lf1(feats)
        out_final = self.lf2(out_final)
        return out_final




