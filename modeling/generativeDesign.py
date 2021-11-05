import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp
from cupyx.scipy import ndimage
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from datetime import datetime
from PIL import Image
import pickle
import os
import sys
import matplotlib.pyplot as plt

# functions
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
        centroids.append((y+75, x+75))
    
    return im, centroids

def get_image(seed):
    im, centroids = generate_window(seed)
    while not len(centroids) > 2:
        im, centroids = generate_window(seed)
    return im, centroids

def extract_features(image, sigma, centroids):

    im_blur = ndimage.gaussian_filter(cp.array(image), sigma=sigma, mode='constant',cval=0)
    im_blur_norm=im_blur*sigma*cp.sqrt(np.pi)
    
    plt.imshow(im_blur_norm.get())
    name = "gaussian_" + str(sigma)+ "_" + str(seed) + ".jpeg"
    plt.savefig(os.path.join(save_dir, name))

    #ha = Image.fromarray(im_blur_norm.get())
    #ha = ha.convert("RGB")
    #ha.save(os.path.join(save_dir, "gauss_blur_" + str(sigma)+ "_" + str(seed) + ".jpeg"))

    im_sx = ndimage.sobel(im_blur_norm, axis=1, mode='reflect')
    im_sy = ndimage.sobel(im_blur_norm, axis=0, mode='reflect')
    im_sobel=np.hypot(im_sx, im_sy)

    plt.imshow(im_sobel.get())
    name = "sobel_" + str(sigma)+ "_" + str(seed) + ".jpeg"
    plt.savefig(os.path.join(save_dir, name))

    #ha = Image.fromarray(im_sobel.get())
    #ha = ha.convert("RGB")
    #ha.save(os.path.join(save_dir, "sobel_filt_" + str(sigma)+ "_" + str(seed) + ".jpeg"))

    feats = []
   
    for centroid in centroids:
        x, y = centroid[0], centroid[1]
        density = cp.mean(im_blur_norm[x-75: x+75, y-75: y+75])
        grad = cp.mean(im_sobel[x-75: x+75, y-75: y+75])
        feats.append([density.get(), grad.get()])

    cp._default_memory_pool.free_all_blocks()

    feats = np.array(feats)
    return feats

def simulate(image, centers, weights):
    test_img = image.copy()
    test_centers = centers.copy()
    
    nx, ny, ix = random_move(test_centers, weights)
    old_x, old_y = test_centers[ix]
    test_img[old_x-75: old_x+75, old_y-75:old_y+75] = 50
    test_img[nx-75:nx+75, ny-75:ny+75] = 255
    test_centers[ix] = (nx, ny)
    
    new_score, _ = evaluate(test_img, test_centers)
    
    return test_img, test_centers, new_score

def random_move(centers, weights):
    found_valid_move = False
    newx, newy, random_index = 0, 0, 0
    
    while not found_valid_move:
        cixs = list(range(len(centers)))
        random_index = 0
        
        if selection == "weighted":
            random_index = np.random.choice(cixs, 1, p = weights)[0] 
        else:
            random_index = np.random.choice(cixs, 1)
        
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
    min_dist = 2*(75*np.sqrt(2)+25)
    valid = np.all(dist > min_dist)
    
    check = cxbool and cybool and valid
    
    return check

def out_of_bounds_check(coord):
    if (coord < 200) or (coord > window_size - 200):
        return False
    else:
        return True

def evaluate(im_pattern, centroids):
    new_feats = []

    for sigma in uniq_sigs:
        feats = extract_features(im_pattern, sigma, centroids)
        new_feats.append(feats[:, 0].reshape(-1,1))
        new_feats.append(feats[:, 1].reshape(-1,1))

    new_feats = [new_feats[ix] for ix in ix_sigs] 
    new_feats = np.hstack(new_feats)
    preds = model.predict(new_feats)
    weights = rel_probs(preds)
    pred_mean = np.mean(preds)
    
    # check if centroids IX matches weights IX
    # should be the case as feats computes features 
    return pred_mean, weights

def rel_probs_exp(arr):
    phiArr = np.exp(-1*arr)
    weights = phiArr / np.sum(phiArr)
    return weights

def rel_probs(arr):
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

if __name__ == '__main__':
    # configs
    curr_dir = os.path.abspath(os.getcwd())
    save_dir = os.path.join(curr_dir, "modeling/disc_gen_outputs")
    niter = 1
    perturb_len = 200
    decay_rate = 0.9993
    pattern_dim = 40
    p = 1/16
    pad = 200
    selection = "weighted"

    # get args from user
    seed = int(sys.argv[1])
    cp.cuda.Device(int(sys.argv[2])).use()

    # save time of experiment
    now = datetime.now()
    log_time = now.strftime("%m_%d_%H%M")

    # save configs
    config_dict = {}
    for variable in ["niter", "perturb_len", "decay_rate", "pad", "selection"]:
        config_dict[variable] = eval(variable)

    a_file = open(os.path.join(save_dir,"configs" + str(seed) + "_" + str(log_time) + ".pkl"), "wb")
    pickle.dump(config_dict, a_file)
    a_file.close()

   # train discriminator
    df = pd.read_csv("all_sigmas_df_comb.csv")
    X, y = df.iloc[:, :-4], df.iloc[:, -1] 
    model = SVR(kernel='rbf')
    X_reduc = backward_elim(X, y, 3)
    model = model.fit(X_reduc, y)

    # get features from backward elimination
    cols = X_reduc.columns.values
    print(cols)
    sigmas = [int(''.join(filter(str.isdigit, item))) for item in X_reduc.columns.values]
    uniq_sigs = sorted(list(set(sigmas)))
    f_density = lambda x: str(x) + "_density"
    f_grad = lambda x: str(x) + "_grad"
    all_feats = [f(sigma) for sigma in uniq_sigs for f in (f_density, f_grad)]
    ix_sigs = [all_feats.index(col) for col in cols]

    # save initial image
    im, centroids  = get_image(seed)
    window_size = len(im)
    im_init = Image.fromarray(im.astype(np.uint8))
    im_init.save(os.path.join(save_dir, "init_seed" + str(seed) + "_" + str(log_time) + ".png"))

    log = []
    # run simulated annealing
    for iter in tqdm(range(niter)):

        curr_score, weights = evaluate(im, centroids)

        new_im, new_centroids, new_score = simulate(im, centroids, weights)

        log.append((curr_score, new_score, centroids))

        if new_score > curr_score:
            im, centroids, curr_score = new_im, new_centroids, new_score

    # save log and final image
    with open(os.path.join(save_dir, "log" + str(seed) + "_" + str(log_time) + ".txt"), "wb") as fp: 
        pickle.dump(log, fp)

    im_final = Image.fromarray(im.astype(np.uint8))
    im_final.save(os.path.join(save_dir, "final_seed" + str(seed) + "_" + str(log_time) + ".png"))