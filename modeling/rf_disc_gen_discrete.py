import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from itertools import product

scale = 200
df = pd.read_csv("dfRandomVer1V4.csv")
subCols = ["density_1000","lap_1000","dipole"]
dfSub = df[subCols]

def get_discriminator(df):
    X, y = df[:, :-1], df[:, -1]
    discriminator = RandomForestRegressor(max_depth=2)
    discriminator.fit(X, y)
    return discriminator

def generate_pattern(pattern_dim):
    threshold = 1/16
    random_pattern = np.random.rand(pattern_dim, pattern_dim)
    binary_pattern = np.where(random_pattern < threshold, 1, 0)
    
    org_locs = np.where(binary_pattern == 1)
        
    return org_locs[0], org_locs[1]
        
def extract_features():
    pad = 6
    window = 1000
    n=0
    pattern_dim = 6

    centroid_x, centroid_y = generate_pattern(pattern_dim)

    cx_scaled = centroid_x + pad
    cy_scaled = centroid_y + pad

    mask_dim = pattern_dim + 2*pad
    mask = np.zeros((mask_dim, mask_dim))
    mask[cx_scaled, cy_scaled] = 1

    obj_exists = mask.flatten()

    coords = [i for i in range(0, mask_dim*scale, scale)]
    coords_scaled = np.array(list(product(coords, coords)))

    non_zero_coords = np.nonzero(obj_exists)

    non_zero_coords = coords_scaled[non_zero_coords]

    feats = []
    for coord in non_zero_coords:
        repeat = np.repeat([coord], len(coords_scaled), axis=0)
        diff = coords_scaled - repeat
        dists = np.sqrt(np.sum(diff**2, axis = 1))

        window_ids = np.argwhere((dists < window) & (dists > 0))

        window_objs = obj_exists[window_ids]
        count = np.sum(window_bool)
        density = count/np.pi*window**2

        window_dists = dists[window_ids]
        weights = (1/window_dists**2)*window_objs - 1
        lap = np.sum(weights)

        diffs_window = diff[window_ids].squeeze()

        grad = weights*diffs_window/window_dists
        grad = np.sum(np.abs(grad), axis = 0)
        grad = np.around(np.sum(grad**2), 10)

        feats.append([coord, grad, density, lap])

    return np.vstack(feats)

def kernel_means(array, kernel_size):
    # find the window with max mean dipole
    
    kernel = np.ones((kernel_size, kernel_size))
    xlim, ylim = array.shape
    xbnd, ybnd = xlim - kernel_size, ylim - kernel_size
    
    best_dipole_mean = array[0:kernel_size, 0:kernel_size].mean()
    best_window = array[0:kernel_size, 0:kernel_size]
    for x in range(xbnd + 1):
        for y in range(ybnd + 1):
            window = array[x:x+kernel_size, y:y+kernel_size]
            dipole_avg = window.mean()
            if dipole_avg > best_dipole_mean:
                best_dipole_mean = dipole_avg
                best_window = window
            
    return best_window, best_dipole_mean

def global_max_generator(iters):
    pattern_dim = 40 
    discriminator = get_discriminator(dfSub.values)
    buffer = []
    iters = iters
    for _ in tqdm(range(iters)):
        centroids_x, centroids_y = generate_pattern(pattern_dim)
        feats = extract_features(centroids_x, centroids_y, pattern_dim)

        mask = np.zeros((pattern_dim, pattern_dim))
        for x,y,f1,f2 in feats:
            feats = np.array([f1, f2]).reshape(1,-1)
            pred = discriminator.predict(feats)
            xscaled = int(x // scale)
            yscaled = int(y // scale)
            mask[xscaled, yscaled] = pred

        best_mask, max_dipole = kernel_means(mask, 6)
        buffer.append((best_mask, max_dipole))
    
    return buffer

