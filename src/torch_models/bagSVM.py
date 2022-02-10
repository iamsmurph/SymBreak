
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import pickle

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tqdm import tqdm
import wandb
import pickle

data_dir = "datasets/round_1/combined/train_cumulative.csv"
df = pd.read_csv("data_dir")
X_train, X_test, y_train, y_test = train_test_split(df.values[:, :-1], df.values[:, -1], test_size=0.1)

# boostrap
n = len(X_train)
bootstraps = [(X_train, y_train)]
for _ in range(1000):
    boot_ix = np.random.choice(n, n, replace=True)
    X_boot = X_train[boot_ix]
    y_boot = y_train[boot_ix]
    bootstraps.append((X_boot, y_boot))

svrs = []
scalers = []
for X, y in bootstraps:
    
    scaler_boot = StandardScaler()
    svr_boot = SVR()
    
    # scale and fit to data
    scaler_boot.fit_transform(X)
    svr_boot.fit(X, y)
    
    # save scaler and fit specific to bootstrap
    scalers.append(scaler_boot)
    svrs.append(svr_boot)

bag_of_preds = []
for i, regr in enumerate(svrs):
    df_cpy = X_test.copy()
    scalers[i].transform(df_cpy)
    preds = regr.predict(df_cpy)
    bag_of_preds.append(preds)

# TO DO: Be able to predict with this model in another file 
#        using the same standard scaler from the original data

if __name__=="__main__":
    pass
