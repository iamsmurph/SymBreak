import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp
from cupyx.scipy import ndimage
from datetime import datetime
from PIL import Image
import pickle
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import bnn
import torch
import multiprocessing
from itertools import repeat

def sim_anneal(im_sim, centroids_sim, niter):
    log = []
    im = im_sim.copy()
    centroids = centroids_sim.copy()

    curr_score, preds, weights = evaluate(im, centroids)
    for iter in tqdm(range(niter)):
        
        new_im, new_centroids, sim_score, preds, new_weights = simulate(im, centroids, weights)

        if sim_score > curr_score:
            im, centroids, curr_score, weights = new_im, new_centroids, sim_score, new_weights
        elif np.random.uniform(0,1) <= random_perturb:
            im, centroids, curr_score, weights = new_im, new_centroids, sim_score, new_weights

        log.append((curr_score, sim_score, preds))
    return im, centroids, log

def simulate(image, centers, weights):
    test_img = image.copy()
    test_centers = centers.copy()
    
    nx, ny, ix = random_move(test_centers, weights)
    old_x, old_y = test_centers[ix]
    test_img[old_x-org_rad: old_x+org_rad, old_y-org_rad:old_y+org_rad] = 50
    test_img[nx-org_rad:nx+org_rad, ny-org_rad:ny+org_rad] = 255
    test_centers[ix] = (nx, ny)
    
    sim_score, preds, weights = evaluate(test_img, test_centers)
    
    return test_img, test_centers, sim_score, preds, weights

def random_move(centers, weights):
    found_valid_move = False
    newx, newy, random_index = 0, 0, 0
    
    while not found_valid_move:
        c_num = len(centers)
        cixs = list(range(c_num))
        
        random_index = np.random.choice(cixs, 1, p=weights)[0]

        angle = np.pi * np.random.uniform(0, 2)
        length = np.rint(max(50, move_len*decay_rate**niter))
        dx, dy = length*np.cos(angle), length*np.sin(angle)
        dx, dy = np.rint(dx), np.rint(dy)
        newx, newy = int(centers[random_index][0] + dx), int(centers[random_index][1] + dy)

        if random_index == 0:
            nbors = centers[1:]
        elif random_index == c_num-1:
            nbors = centers[:-1]
        else:
            nbors = centers[:random_index] + centers[random_index+1:]
        
        found_valid_move = validate(newx, newy, nbors)
    return newx, newy, random_index

def validate(cx, cy, centroids):   
    cxbool = out_of_bounds_check(cx)
    cybool = out_of_bounds_check(cy)
    
    rep = np.tile(np.array([cx, cy]).reshape(-1,2), [len(centroids), 1])
    centroids_arr = np.array(centroids)
    dist = np.sqrt(np.sum((rep-centroids_arr)**2, axis = 1))
    min_dist = 2*(org_rad+org_pad)
    valid = np.all(dist > min_dist)
    
    check = cxbool and cybool and valid
    
    return check

def out_of_bounds_check(coord):
    if (coord < 200) or (coord > size - 200):
        return False
    else:
        return True

def extract_features(image, sigma, centroids):

    im_blur = ndimage.gaussian_filter(cp.array(image), sigma=sigma, mode='constant',cval=0)
    im_blur_norm=im_blur*sigma*cp.sqrt(np.pi)

    im_sx = ndimage.sobel(im_blur_norm, axis=1, mode='reflect')
    im_sy = ndimage.sobel(im_blur_norm, axis=0, mode='reflect')
    im_sobel=np.hypot(im_sx, im_sy)

    feats = []
   
    for centroid in centroids:
        x, y = centroid[0], centroid[1]
        density = np.mean(im_blur_norm[x-org_rad: x+org_rad, y-org_rad: y+org_rad])
        if np.isnan(density):
            print(x,y)
        assert(~np.isnan(density))
        grad = np.mean(im_sobel[x-org_rad: x+org_rad, y-org_rad: y+org_rad])
        assert(~np.isnan(grad))
        feats.append([density.get(), grad.get()])

    cp._default_memory_pool.free_all_blocks()

    feats = np.array(feats)
    return feats
    
def get_weights(arr, weights):
    if weights == "uniform":
        n = len(arr)
        return [1/n]*len(arr)
    elif weights == "exponential":
        phiArr = np.exp(-1*arr)
        weights = phiArr / np.sum(phiArr)
    else:
        maxVal, minVal = np.max(arr), np.min(arr)
        phiArr = -1*arr+minVal+maxVal
        weights = phiArr / np.sum(phiArr)
        return weights

def evaluate(im_pattern, centroids, weights = "uniform"):
    new_feats = []

    #new_feats = pool_obj.map(extract_features,)
    for i, sigma in enumerate(sigmas):
        
        feats = extract_features(im_pattern, sigma, centroids)
        new_feats.append(feats[:, 0].reshape(-1,1))
        new_feats.append(feats[:, 1].reshape(-1,1))

    new_feats = scaler.transform(np.hstack(new_feats))
    new_feats = torch.tensor(new_feats).float()
    preds = model.forward(new_feats).detach().numpy()
    weights = get_weights(preds, weights)
    pred_mean = np.mean(preds)
    #print(pred_mean)
    
    return pred_mean, preds, weights
def generate_image(num_organoids, search_iter):
    # create empty pattern
    mask = np.zeros((size, size))
    centroids = []

    # initialize with ten organoids
    for _ in range(init_num_organoids):
        x, y = random_location(mask, size)
        centroids.append((x,y))
        mask[x-org_rad:x+org_rad, y-org_rad:y+org_rad] = 255
    #score, _, _ = evaluate(mask, centroids)

    # stochastically search for good organoids
    for i in tqdm(range(num_organoids)):
        sim_organoids = []
        for _ in range(search_iter):
            # copy to evaluate independently
            test_mask = mask.copy()
            test_centroids = centroids.copy()

            # sample test location in mask
            x, y = random_location(mask, size)
            test_centroids.append((x,y))
            test_mask[x-org_rad:x+org_rad, y-org_rad:y+org_rad] = 255
            _, scores, _ = evaluate(test_mask, test_centroids)
            # get the score of the test organoid 
            test_score = scores[-1]
            sim_organoids.append((test_score, x, y))
        
        # select best simulated organoid
        best_sim_organoid = sorted(sim_organoids)[-1]
        newx, newy = best_sim_organoid[1], best_sim_organoid[2]
        centroids.append((newx, newy))
        mask[newx-org_rad:newx+org_rad, newy-org_rad:newy+org_rad] = 255

    # add the pad at the end
    gen_pattern = np.pad(mask, pad)
    gen_centroids = np.array(centroids) + pad

    return gen_pattern, gen_centroids

def random_location(image, size):
    found_valid = False
    
    while not found_valid:
        x, y = np.random.choice(size,2)
        min_space = org_pad*2
        xl, xr, yu, yd = x-org_rad, x+org_rad, y-org_rad, y+org_rad
        # check if within bounds
        if xl > 0+min_space and yu > 0+min_space and xr < size-min_space and yd < size-min_space:
            new_loc = image[xl-min_space:xr+min_space, yu-min_space:yd+min_space]
            # check if overlapping with organoid
            if np.sum(new_loc) == 0: 
                found_valid = True
    return x,y

if __name__ == '__main__':
    # configs
    curr_dir = os.path.abspath(os.getcwd())
    save_dir = os.path.join(curr_dir, "disc_gen_outputs")
    niter = 5000
    move_len = 300
    decay_rate = 0.99965
    #pattern_dim = 40
    #p = 1/16
    size = 6000
    pad = 200
    org_rad = 75
    org_pad = 25
    init_num_organoids = 10
    n_added_orgs = 90
    n_search = 10
    random_perturb = 1/16
    
 
    sigmas = list(range(200,1200,200))
    
    # load saved model and scaler
    scaler = pickle.load(open("modeling/torch_models/bnn_scaler.pkl", "rb"))
    model = bnn.BayesianRegressor(10,1)
    model.load_state_dict(torch.load('modeling/torch_models/bnn_model'))
    model.eval()

    #pool_obj = multiprocessing.Pool()
    # GET GPU INDEX FROM USER VIA TERMINAL ARGUMENT
    cp.cuda.Device(int(sys.argv[1])).use()

    # image generation
    im, centroids = generate_image(n_added_orgs, n_search)
    im_final = Image.fromarray(im.astype(np.uint8))
    im_final.save(os.path.join(save_dir, "init_mask.png"))
    np.save(os.path.join(save_dir, "gen_image_centroids"), centroids)

    new_im, new_centroids, log = sim_anneal(im, centroids.tolist(), niter)
    n_im = Image.fromarray(new_im)
    n_im = n_im.convert("L")
    n_im.save(os.path.join(save_dir, "anneal_image.png"))
    np.save(os.path.join(save_dir, "anneal_image_centroids"), new_centroids)
    pickle.dump(log, open(os.path.join(save_dir, "anneal_log.txt"), "wb"))