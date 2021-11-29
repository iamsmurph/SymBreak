import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp
from cupyx.scipy import ndimage
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from datetime import datetime
from PIL import Image
import pickle
import os
import sys
import matplotlib.pyplot as plt

def generate_window(seed):
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
        centroids.append((y+org_rad, x+org_rad))
    
    return im, centroids

def get_image(seed):
    im, centroids = generate_window(seed)
    while not len(centroids) > 2:
        im, centroids = generate_window(seed)
    return im, centroids

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
        cixs = list(range(len(centers)))
        
        random_index = np.random.choice(cixs, 1, p=weights)[0]

        angle = np.pi * np.random.uniform(0, 2)
        length = np.rint(max(50, perturb_len*decay_rate**iter))
        dx, dy = length*np.cos(angle), length*np.sin(angle)
        dx, dy = np.rint(dx), np.rint(dy)
        newx, newy = int(centers[random_index][0] + dx), int(centers[random_index][1] + dy)

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
    if (coord < 200) or (coord > window_size - 200):
        return False
    else:
        return True

def evaluate(im_pattern, centroids, weights = "uniform"):
    new_feats = []

    for sigma in uniq_sigs:
        feats = extract_features(im_pattern, sigma, centroids)
        new_feats.append(feats[:, 0].reshape(-1,1))
        new_feats.append(feats[:, 1].reshape(-1,1))

    new_feats = [new_feats[ix] for ix in ix_sigs] 
    #print(new_feats)
    new_feats = scaler.transform(np.hstack(new_feats))
    preds = model.predict(new_feats)
    weights = get_weights(preds, weights)
    pred_mean = np.mean(preds)
    #print(pred_mean)
    
    return pred_mean, preds, weights

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

def backward_elim(X_iter, y, n_feats):
    n_iter = X_iter.shape[1]
    
    while n_iter > n_feats:
        feat_ix = list(range(X_iter.shape[1]))
    
        min_ix = 0
        min_error = -np.infty
        for ix in feat_ix:
            cols = feat_ix[:ix] + feat_ix[ix+1:]
            temp_df = X_iter.iloc[:, cols]
            temp_error = cross_val_score(model, temp_df, y, cv = 5, scoring = "neg_mean_squared_error").mean()
            if temp_error > min_error:
                min_ix = ix
                min_error = temp_error

        cols = feat_ix[:min_ix] + feat_ix[min_ix+1:]
        X_iter = X_iter.iloc[:, cols]
        n_iter = X_iter.shape[1]
        
    return X_iter

def generate_image(num_organoids, search_iter):
    # create empty pattern
    mask = np.zeros((size, size))
    centroids = []

    # initialize with ten organoids
    for _ in range(10):
        x, y = random_location(mask, size)
        centroids.append((x,y))
        mask[x-org_rad:x+org_rad, y-org_rad:y+org_rad] = 255
    #score, _, _ = evaluate(mask, centroids)

    im_final = Image.fromarray(mask.astype(np.uint8))
    im_final.save(os.path.join(save_dir, "init_mask.png"))

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
        if xl > 0 and yu > 0 and xr < size and yd < size:
            new_loc = image[xl-min_space:xr+min_space, yu-min_space:yd+min_space]
            # check if overlapping with organoid
            if np.sum(new_loc) == 0: 
                found_valid = True
    return x,y

def sim_anneal(im, centroids, niter):
    log = []
    curr_score, preds, weights = evaluate(im, centroids)
    for iter in tqdm(range(niter)):
        
        new_im, new_centroids, sim_score, preds, new_weights = simulate(im, centroids, weights)

        if sim_score > curr_score:
            im, centroids, curr_score, weights = new_im, new_centroids, sim_score, new_weights
        elif np.random.uniform(0,1) <= random_perturb:
            im, centroids, curr_score, weights = new_im, new_centroids, sim_score, new_weights

        log.append((curr_score, sim_score, centroids, preds))
    return log

if __name__ == '__main__':
    # configs
    curr_dir = os.path.abspath(os.getcwd())
    save_dir = os.path.join(curr_dir, "modeling/disc_gen_outputs")
    niter = 3000
    perturb_len = 200
    decay_rate = 0.99954
    pattern_dim = 40
    p = 1/16
    size = 8000
    pad = 200
    org_rad = 75
    org_pad = 25
    n_orgs = 100
    n_search = 10
    random_perturb = 1/16

    # get args from user
    #seed = int(sys.argv[1])
    cp.cuda.Device(int(sys.argv[1])).use()

    '''
    # save time of experiment
    now = datetime.now()
    log_time = now.strftime("%m_%d_%H%M")

    # save configs
    config_dict = {}
    for variable in ["niter", "perturb_len", "decay_rate", "pad", "random_perturb"]:
        config_dict[variable] = eval(variable)
    pickle.dump(config_dict, open(os.path.join(save_dir,"configs" + str(seed) + "_" + str(log_time) + ".pkl"), "wb"))
    '''

   # feature selection
    big_scaler = RobustScaler()
    df = pd.read_csv("all_sigmas_df_comb.csv")
    df_scaled = big_scaler.fit_transform(df.iloc[:, :-4])
    X = pd.DataFrame(df_scaled)
    X.columns = df.iloc[:, :-4].columns.values
    y = df.iloc[:, -1] 
    model = SVR(kernel='rbf')
    X_reduce = backward_elim(X, y, 1)
    
    # train discriminator
    X_train = df[X_reduce.columns.values]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    model = model.fit(X_train, y)

    # get features from backward elimination
    cols = X_reduce.columns.values
    sigmas = [int(''.join(filter(str.isdigit, item))) for item in X_reduce.columns.values]
    uniq_sigs = sorted(list(set(sigmas)))
    f_density = lambda x: str(x) + "_density"
    f_grad = lambda x: str(x) + "_grad"
    all_feats = [f(sigma) for sigma in uniq_sigs for f in (f_density, f_grad)]
    ix_sigs = [all_feats.index(col) for col in cols]


    #****************************
    # save initial image
    #im, centroids  = get_image(seed)
    #window_size = len(im)
    #im_init = Image.fromarray(im.astype(np.uint8))
    #im_init.save(os.path.join(save_dir, "init_seed" + str(seed) + "_" + str(log_time) + ".png"))

    # run simulated anneal
    im, centroids = generate_image(n_orgs, n_search)
    plt.imshow(im)
    plt.show()

    im_final = Image.fromarray(im.astype(np.uint8))
    im_final.save(os.path.join(save_dir, "test.png"))

    '''
    window_size = len(im)
    log = sim_anneal(im, centroids, niter)


    #****************************


    # save log and final image
    pickle.dump(log, open(os.path.join(save_dir, "log" + str(seed) + "_" + str(log_time) + ".txt"), "wb"))

    im_final = Image.fromarray(im.astype(np.uint8))
    im_final.save(os.path.join(save_dir, "final_seed" + str(seed) + "_" + str(log_time) + ".png"))
    '''