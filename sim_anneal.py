import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp
from cupyx.scipy import ndimage
from sklearn.svm import SVR
import random
import time
from PIL import Image
import pickle

def extract_features(image, sigma, centroids):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    im_blur = ndimage.gaussian_filter(cp.array(image), sigma=sigma, mode='constant',cval=0)
    
    im_blur_norm=im_blur*sigma*cp.sqrt(np.pi)

    im_sx = ndimage.sobel(im_blur_norm, axis=1, mode='reflect')
    im_sy = ndimage.sobel(im_blur_norm, axis=0, mode='reflect')
    im_sobel=np.hypot(im_sx, im_sy)

    feats = []
   
    for centroid in centroids:
        x, y = centroid[0], centroid[1]
        density = cp.nanmean(im_blur_norm[x-75: x+75, y-75: y+75])
        grad = cp.nanmean(im_sobel[x-75: x+75, y-75: y+75])
        feats.append([density.get(), grad.get()])

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    feats = np.array(feats)
    return feats


def generate_window(window_size, p, pad, seed):
    np.random.seed(seed)
    start = time.time()
    random_pattern = np.random.rand(window_size, window_size)
    binary_pattern = np.where(random_pattern < p, 1, 0)

    org_locs = np.argwhere(binary_pattern == 1)

    org_locs_scaled = org_locs*200+pad
    pattern_dim_scaled = window_size*200+2*pad
    
    centroids = []
    im = np.zeros((pattern_dim_scaled, pattern_dim_scaled))

    for y, x in org_locs_scaled:
        im[y:y+150,x:x+150] = 255
        centroids.append((y+75, x+75))
    
    return im, centroids, pattern_dim_scaled

def simulate(image, centers, window_size, std):
    nx, ny, ix = random_move(centers, window_size, std)
    
    # make new one
    test_img = np.copy(image)
    test_centers = centers.copy()
    old_x, old_y = test_centers[ix]
    test_img[old_x-75: old_x+75, old_y-75:old_y+75] = 0
    test_img[nx-75:nx+75, ny-75:ny+75] = 255
    test_centers[ix] = (nx, ny)
    
    new_score, feat_values = evaluate(test_img, test_centers)
    
    return test_img, test_centers, new_score, feat_values

def validate(cx, cy, centroids, window_size):    
    rep = np.tile(np.array([cx, cy]).reshape(-1,2), [len(centroids), 1])
    centroids_arr = np.array(centroids)
    dist = np.sqrt(np.sum((rep-centroids_arr)**2, axis = 1))
    min_dist = 2*(75*np.sqrt(2)+25)
    valid = np.all(dist > min_dist)
    
    return valid

def evaluate(im_pattern, centroids):
    feats400 = extract_features(im_pattern, 400, centroids)
    
    if len(feats400) > 0:
        density = feats400[:, 0].reshape(-1,1)
        grad = feats400[:, 1].reshape(-1,1)
        newX = np.hstack((density, grad))
        preds = rbf_svr_fit.predict(newX)
        pred_mean = np.mean(preds)
        
    return pred_mean, newX

def random_move(centers, window_size, std):
    found_valid_move = False
    newx, newy, random_index = 0, 0, 0
    
    while not found_valid_move:
        random_index = random.randint(0,len(centroids)-1)

        dx, dy = np.random.normal(0, std, 2)
        dx, dy = np.rint(dx), np.rint(dy)
        
        newx, newy = int(centers[random_index][0] + dx), int(centers[random_index][1] + dy)
        newx = out_of_bounds_check(newx, window_size)
        newy = out_of_bounds_check(newy, window_size)

        nbors = centers[:random_index] + centers[random_index+1:]
        
        found_valid_move = validate(newx, newy, nbors, window_size)
    
    return newx, newy, random_index

def out_of_bounds_check(coord, window_size):
    if coord < 100 + 75:
        coord = 175
        return coord
    elif coord > window_size - 100 - 75:
        coord = window_size - 100 - 75
        return coord
    else:
        return coord
        

def get_image(window_size, p, pad, seed):
    im, centroids, window_size = generate_window(window_size, p, pad, seed)
    while not len(centroids) > 2:
        im, centroids, window_size = generate_window(window_size, p, pad, seed)
    return im, centroids, window_size


def p_accept(t, beta):
    return t/(t+t*beta)

seeds = [3, 4, 7, 8, 11, 12]
log = []
niter = 1500
std_perturb = 50
epsilon = 0
pattern_dim = 40
p = 1/16
pad = 100

for seed in seeds:
    # save initial image
    im, centroids, window_size = get_image(pattern_dim, p, pad, seed)
    im_init = Image.fromarray(im.astype(np.uint8))
    im_init.save("init_seed_" + str(seed) + ".png")

    # train discriminator
    dfHeitorRandom = pd.read_csv("random_features_dipole_v3.csv")
    rX, ry = dfHeitorRandom.iloc[:, :-1], dfHeitorRandom.iloc[:, -1]
    rxSub = rX[["density400", "grad400"]]
    rbf_svr = SVR(kernel='rbf')
    rbf_svr_fit = rbf_svr.fit(rxSub, ry.ravel())

    # run simulated annealing
    for i in tqdm(range(1, niter+1)):

        curr_score, old_feats = evaluate(im, centroids)

        new_im, new_centroids, new_score, new_feats = simulate(im, centroids, window_size, std_perturb)

        log.append((curr_score, old_feats, new_score, new_feats, centroids))

        if new_score - curr_score > epsilon:
            im, centroids, curr_score = new_im, new_centroids, new_score

    # save log and final image
    with open("log" + str(seed) + ".txt", "wb") as fp: 
        pickle.dump(log, fp)

    im_final = Image.fromarray(im.astype(np.uint8))
    im_final.save("final_seed_" + str(seed)+ ".png")