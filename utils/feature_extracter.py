import numpy as np
import pandas as pd
import cupy as cp
from cupyx.scipy import ndimage
import pickle
from tqdm import tqdm

# version with organoid left out 
def one_out_mask(out_ix, centroids, x_size, y_size, org_rad = 75):
    mask = np.zeros((x_size, y_size))
    for i, (c1, c2) in enumerate(centroids):
        if i != out_ix:
            x = int(c2)
            y = int(c1)
            mask[x-org_rad:x+org_rad, y-org_rad:y+org_rad] = 255
    return mask

def max_gradient(localIm, org_rad, name=None):
    x, y = 75, 75 

    xmax = localIm[x + org_rad - 1, y]
    xmin = localIm[x - org_rad, y]
    ymax = localIm[x, y + org_rad - 1]
    ymin = localIm[x, y - org_rad]

    xdiffnorm  = (xmax - xmin)/(2*org_rad)
    ydiffnorm = (ymax - ymin)/(2*org_rad)

    grad = np.sqrt(xdiffnorm**2 + ydiffnorm**2)
    gradVec = np.array((xdiffnorm, ydiffnorm))
                   
    return grad, gradVec

def extract_features(sigma, centroids, pad=500, org_rad=75):
    
    max_coord = int(np.max(centroids))
    min_coord = int(np.min(centroids))
    
    x_size, y_size = max_coord + min_coord + pad*2, max_coord + min_coord + pad*2
    
    dCentroids_scaled = centroids+pad
    
    grads = []
    gradVecs = []
    for i, (c1, c2) in enumerate(dCentroids_scaled):
        x = int(c2)
        y = int(c1)
        one_out = one_out_mask(i, dCentroids_scaled, x_size, y_size)
        
        im = cp.array(one_out)
        im_blur = ndimage.gaussian_filter(im, sigma=sigma, mode = 'constant')

        dfeats = im_blur[x-org_rad:x+org_rad, y-org_rad:y+org_rad]
        
        grad, gradVec = max_gradient(dfeats.get(), org_rad)
        
        grads.append(grad)
        gradVecs.append(gradVec)
      
    return grads, gradVecs

if __name__ == "__main__":
    df_round1_1 = pd.read_csv("datasets/round_1/combined/big_df1.csv")
    df_round1_2 = pd.read_csv("datasets/round_1/combined/big_df2.csv")
    df_round1_3 = pd.read_csv("datasets/round_1/combined/big_df3.csv")
    df_round1_4 = pd.read_csv("datasets/round_1/combined/big_df4.csv")
    df_round1_5 = pd.read_csv("datasets/round_1/combined/big_df5.csv")
    df_round1_6 = pd.read_csv("datasets/round_1/combined/big_df6.csv")
    designs = [df_round1_1, df_round1_2, df_round1_3, df_round1_4, df_round1_5, df_round1_6]

    sigmas = [100, 200, 300, 400, 500]

    round1_grads = []
    round1_gradVecs = []
    for sigma in tqdm(sigmas):
        sigma_grads = []
        sigma_gradVecs = []
        for df in designs: 
            coords = np.vstack([df.dx.values, df.dy.values]).T
            grads, gradVecs = extract_features(sigma, coords)
            sigma_grads.append(grads)
            sigma_gradVecs.append(gradVecs)
        round1_grads.append(sigma_grads)
        round1_gradVecs.append(sigma_gradVecs)

    pickle.dump(round1_grads, open("round1_grads.txt", "wb"))

    pickle.dump(round1_gradVecs, open("round1_gradVecs.txt", "wb"))