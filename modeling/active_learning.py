import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate Images of Selected "Uncertain" Datasets

def generate_window(seed, p, pattern_dim = 40, pad = 200):
    np.random.seed(seed)
    random_pattern = np.random.rand(pattern_dim, pattern_dim)
    binary_pattern = np.where(random_pattern < p, 1, 0)

    org_locs = np.argwhere(binary_pattern == 1)

    org_locs_scaled = org_locs*200+pad
    pattern_dim_scaled = pattern_dim*200+2*pad
    
    centroids = []
    im = np.zeros((pattern_dim_scaled, pattern_dim_scaled))

    for y, x in org_locs_scaled:
        im[y:y+150,x:x+150] = 255
        centroids.append((y+75, x+75))
    
    return im, centroids

unlabeled_df = pd.read_csv("unlabeled_df.csv")

rank = np.loadtxt("bnn_blitz_dfRanks.txt", delimiter=',')

half = len(rank)//2

badDFs_ixs = rank[half:][:, 0].astype(int)

def get_mean_feats(ixs):
    mean_feats = []
    for ix in ixs:
        X = unlabeled_df[unlabeled_df.dataset == ix].iloc[:, :10].values
        means = np.mean(X, axis = 0)
        mean_feats.append(means)
    return mean_feats

points = np.array(get_mean_feats(badDFs_ixs))

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering().fit(points)

n = len(np.unique(clustering.labels_))

# limit to 6 clusters maximum
ndata_per_clus = 6 // n

ctr = 0
clus_choices = []
for i in range(n):
    ixs = np.where(clustering.labels_ == i)[0]
    dataset_ixs = rank[ixs+half][:, 0].astype(int)
    if n-1 == i:
        ndata_per_clus = 6 - ctr
    
    chosen_ixs = np.random.choice(dataset_ixs, ndata_per_clus, replace = True)
    clus_choices.append(chosen_ixs)
    ctr += ndata_per_clus
    

clus_choices

df = unlabeled_df[unlabeled_df.dataset == 17]
p, seed = np.unique(df.p)[0], np.unique(df.seed)[0] 
print(p,seed)

nxt_rnd_patts = []
for clus in clus_choices:
    for ix in clus:
        df = unlabeled_df[unlabeled_df.dataset == ix]
        p, seed = np.unique(df.p)[0], np.unique(df.seed)[0] 
        im, centroids = generate_window(seed, p)
        print(p, seed, len(centroids))
        nxt_rnd_patts.append((centroids, seed, p))
        plt.figure(figsize=(10,10))
        plt.imshow(im)
        plt.show()