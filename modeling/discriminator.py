import torch
from torch.functional import Tensor
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.model_selection import train_test_split
import wandb

data_dir = "datasets/dfExpertFeatRandomVer1_Model.npy"

hparam = {"num_epochs": 10, "batch_size": 60, "lr": 1e-4, "dropout_prob": 0.25, 
            "valProp": .2, "trShuffle": True, "tstShuffle": False}

class DiscriminatorDataset(Dataset):
    """ Define dataset operations """
    def __init__(self, X, y):
        self.n_samples = X.shape[0]
        
        x1 = X[:, :3].astype(np.float32)
        x2 = X[:, 3:].astype(np.float32)

        self.x1_data = torch.from_numpy(x1)
        self.x2_data = torch.from_numpy(x2)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        sample = self.x1_data[index], self.x2_data[index], self.y_data[index]
        return sample

    def __len__(self):
        return self.n_samples

class ComplementLayer(nn.Module):
    """ Custom layer for ensuring complementary probabilities """
    def __init__(self, in_features, out_features):
        super(ComplementLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        assert(self.in_features % 2 == 0)
        assert(self.out_features % 2 == 0)

        self.weights = nn.Parameter(torch.empty(out_features, in_features // 2).to(device), requires_grad = True)
        
        self.bias = nn.Parameter(torch.empty(1, out_features // 2).to(device), requires_grad = True)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))            
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weights_flip = torch.flip(self.weights, [0,1])
        weights_cat = torch.cat((self.weights, weights_flip), 1)

        bias_flip = torch.flip(self.bias, [0])
        bias_cat = torch.cat((self.bias, bias_flip), 1)
        
        w_times_x = torch.mm(x, weights_cat.t())
        return torch.add(w_times_x, bias_cat)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hparam["dropout_prob"])
        self.softmax = nn.Softmax()
        
        # for feature extraction model
        self.l1 = nn.Linear(3, 10, bias=True)
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
        feats = torch.cat((out1, out2_flip), 1)
        
        out_final = self.lf1(feats)
        out_final = self.relu(out_final)
        out_final = self.dropout(out_final)
        out_final = self.lf2(out_final)
        return out_final

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset preparation
df = np.load(data_dir)
X = df[:, :-1].astype(np.float32)

TargClasses = df[:, -1]
y = TargClasses.astype(np.int_)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=hparam["valProp"])

trainDataset = DiscriminatorDataset(X_train, y_train)
valDataset = DiscriminatorDataset(X_val, y_val)
nTrain = len(trainDataset)

train_loader = DataLoader(dataset = trainDataset,
                          batch_size = hparam["batch_size"],
                          shuffle = hparam['trShuffle'],
                          num_workers = 2)

val_loader = DataLoader(dataset = valDataset,
                          batch_size = hparam["batch_size"],
                          shuffle = hparam['tstShuffle'],
                          num_workers = 2)

# checking if data loader and forward model looks correct


model = Discriminator().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hparam["lr"])

# Train the model
for epoch in range(hparam['num_epochs']):
    for i, (x1, x2, label) in enumerate(train_loader):  
        
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.to(device)
        
        # Forward pass
        outputs = model(x1, x2)

        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
