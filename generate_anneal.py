import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp
from cupyx.scipy import ndimage
import pickle
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
import wandb

def max_gradient(x,y, im_blur):

    # get pixel intensities in extremes
    xmax = im_blur[x + run.config.org_rad - 1, y]
    xmin = im_blur[x - run.config.org_rad, y]
    ymax = im_blur[x, y + run.config.org_rad - 1]
    ymin = im_blur[x, y - run.config.org_rad]

    xdiffnorm  = ((xmax - xmin) / run.config.org_rad).get()
    ydiffnorm = ((ymax - ymin) / run.config.org_rad).get()

    # gradient magnitude and vector
    grad = np.array(np.sqrt(xdiffnorm**2 + ydiffnorm**2))
    gradVec = np.array((xdiffnorm, ydiffnorm))
                   
    return grad, gradVec


def extract_features(img, sigma, centroids):
    # gaussian blur
    im = cp.array(img)
    im_blur = ndimage.gaussian_filter(im, sigma=sigma, mode = 'constant')
    
    # extract features per centroid
    feats = []
    for _, (x, y) in enumerate(centroids):
        
        bool_mat = draw_circle(x,y, img, run.config.org_rad)
        bool_mat_gpu = cp.array(bool_mat) 

        dfeats = im_blur[bool_mat_gpu]
        density = cp.mean(dfeats).get()
        grad, _ = max_gradient(x,y, im_blur)
        feats.append((density, grad))
      
    return np.array(feats)

def sim_anneal(im, centroids):
    #print("Starting simulated annealing...")
    log = []

    curr_score, preds, feats = evaluate(im, centroids)
    log.append([preds, feats])

    for i in tqdm(range(run.config.niter)):
        # get candidate perturbation
        new_im, new_centroids, sim_score, preds, newfeats = candidate(im, centroids, i)
        
        # determine random acceptance threshold
        p = np.random.uniform(0,1)
        p_perturb = run.config.random_perturb*run.config.perturb_decay**(i+1)

        # accept if good score or if passes threshold
        if  sim_score > curr_score or p < p_perturb:
            im, centroids, curr_score, feats = new_im, new_centroids, sim_score, newfeats

        log.append([preds, feats])
        wandb.log({"curr_score": curr_score})
        if i % run.config.snapshot_step == 0:
            name = "anneal_step_%d" % i
            wandb.log({name: wandb.Image(im.astype(np.uint8))})
    return im, centroids, log

def candidate(image, centers, ix):
    test_img = image.copy()
    test_centers = centers.copy()
    im_size = len(image)
    assert(run.config.size < im_size)
    
    nx, ny, ix = random_move(test_centers, im_size, ix)

    old_x, old_y = test_centers[ix]
    # erase
    test_img = draw_circle(old_x, old_y, test_img, run.config.org_rad, fill=True, fillVal=0)
    # fill new
    test_img = draw_circle(nx, ny, test_img, run.config.org_rad, fill=True)
    #test_img[old_x-75:old_x+75, old_y-75:old_y+75] = 50
    #test_img[nx-75:nx+75, ny-75:ny+75] = 255

    test_centers[ix] = (nx, ny)
    sim_score, preds, newfeats = evaluate(test_img, test_centers)
    
    return test_img, test_centers, sim_score, preds, newfeats

def random_move(centers, im_size, ix):
    found_valid_move = False
    newx, newy, random_index = 0, 0, 0
    
    while not found_valid_move:
        c_num = len(centers)
        cixs = list(range(c_num))
        
        # randomly choose organoid to be perturbed
        random_index = np.random.choice(cixs)

        # determine random x, y perturbation
        angle = np.pi * np.random.uniform(0, 2)
        length = run.config.move_len*run.config.move_decay**(ix+1)
        dx, dy = length*np.cos(angle), length*np.sin(angle)
        dx, dy = np.rint(dx), np.rint(dy)
        newx, newy = int(centers[random_index][0] + dx), int(centers[random_index][1] + dy)

        # get the neighbors of chosen organoid
        if random_index == 0:
            nbors = centers[1:]
        elif random_index == c_num-1:
            nbors = centers[:-1]
        else:
            nbors = centers[:random_index] + centers[random_index+1:]
        
        found_valid_move = validate(newx, newy, nbors, im_size)
    return newx, newy, random_index

def validate(cx, cy, centroids, im_size):
    # check if out of bounds   
    cxbool = out_of_bounds_check(cx, im_size)
    cybool = out_of_bounds_check(cy, im_size)
    
    # check centroid to centroid distance
    rep = np.tile(np.array([cx, cy]).reshape(-1,2), [len(centroids), 1])
    centroids_arr = np.array(centroids)
    dist = np.sqrt(np.sum((rep-centroids_arr)**2, axis = 1))
    valid = np.all(dist > run.config.c_to_c_dist)
    
    check = cxbool and cybool and valid

    return check

def out_of_bounds_check(coord, im_size):
    if (coord < run.config.min_dist) or (coord > im_size - run.config.min_dist):
        return False
    else:
        return True
    
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

def evaluate(im_pattern, centroids):
    new_feats = []
    
    for sigma in model_sigmas:
        feats = extract_features(im_pattern, sigma, centroids)
        new_feats.append(feats[:, 0].reshape(-1,1))
        new_feats.append(feats[:, 1].reshape(-1,1))

    new_feats = np.hstack(new_feats)
    model_feats = new_feats[:, feats_ixs]
    model_feats_scaled = scaler.transform(model_feats)
    preds = model.predict(model_feats_scaled)

    reward = reward_fn(preds, run.config.objective, run.config.lmbda)

    return reward, preds, model_feats_scaled

def reward_fn(preds, objective, lmbda = None):
    if objective == "min":
        return np.min(preds)
    elif objective == "mean":
        return np.mean(preds)
    else:
        assert(lmbda != None)
        return np.mean(preds) + lmbda*np.min(preds)
    

def generate_image():
   # print("Starting stochastic pattern generation...")
    # create empty pattern
    mask = np.zeros((run.config.size, run.config.size))
    centroids = []

    # Initialize mask
    for _ in range(run.config.n_init_orgs):
        x, y = random_location(mask)
        centroids.append((x,y))
        mask = draw_circle(x,y, mask, run.config.org_rad, fill=True)
    
    # Sequentially add organoids stochastically
    for _ in tqdm(range(run.config.n_total_orgs-1)):
        sim_organoids = []
        for _ in tqdm(range(run.config.n_search), leave = False):
            # copy to evaluate independently
            test_mask = mask.copy()
            test_cents = centroids.copy()

            # sample test location in mask
            x, y = random_location(mask)
            test_cents.append((x,y))
            test_mask = draw_circle(x,y, test_mask, run.config.org_rad, fill=True)
            score, _, _ = evaluate(test_mask, test_cents)
            sim_organoids.append((score, x, y))
        
        # select best simulated organoid
        best_sim_organoid = sorted(sim_organoids)[-1]
        newx, newy = best_sim_organoid[1], best_sim_organoid[2]
        mask = draw_circle(newx, newy, mask, run.config.org_rad, fill = True)
        centroids.append((newx, newy))

    # add the pad at the end
    gen_pattern = np.pad(mask, run.config.pad)
    gen_centroids = np.array(centroids) + run.config.pad

    return gen_pattern, gen_centroids

def random_location(image):
    found_valid = False
    
    while not found_valid:
        x, y = np.random.choice(run.config.size, 2)
        # ensure organoid is at least 2 pads worth away from wall and organoids
        if (x > run.config.min_dist and x < run.config.size-run.config.min_dist and 
            y > run.config.min_dist and y < run.config.size-run.config.min_dist):
            bool_mat = draw_circle(x, y, image, run.config.min_dist)
            
            # check if overlapping with other organoids
            if np.sum(image[bool_mat]) == 0: 
                found_valid = True
    return x,y

def draw_circle(x,y, orgIm, radius, fill = False, fillVal = 255):
    dim = len(orgIm)
    xx, yy = np.mgrid[:radius*2, :radius*2]
    zz = (xx - radius) ** 2 + (yy - radius) ** 2
    circle = zz < radius ** 2
    
    bool_mat = np.pad(circle, ((x-radius, dim-x-radius),(y-radius, dim-y-radius)))
    if fill:
        orgIm[bool_mat] = fillVal
        return orgIm
    else:
        return bool_mat

if __name__ == '__main__':

    config = dict(save_dir = os.path.join(os.path.abspath(os.getcwd()), "gen_anneal_outputs"), 
                    pad = 500,
                    org_rad = 75,
                    org_pad = 25,
                    snapshot_step = 100,
                    min_dist = 75+2*1,
                    c_to_c_dist = 2*75+2*1,
                    objective = "min",
                    lmbda = None,
                    # seq gen configs
                    n_init_orgs = 1, 
                    n_total_orgs = 8,
                    n_search = 30,
                    size = 2000, 
                    # sim anneal configs
                    niter = 2000,
                    move_len = 80,
                    move_decay = .9991,
                    random_perturb = 1/5,
                    perturb_decay = .9977) 

    # Get GPU index from user
    cp.cuda.Device(int(sys.argv[1])).use()

    # load model and scaler
    model_sigmas = [200, 700]
    feats_ixs = np.array([0,1,1,0]).astype(np.bool8)
    combined_df = pd.read_csv("datasets/round_1/combined/circle_combined_df.csv")
    feats_df = combined_df[['grad200','density700']]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(feats_df.values)
    y = combined_df.cdx2Dipole.values
    model = KernelRidge(kernel="rbf").fit(X, y)

    for n_orgs in [5,6,7]: 
        for i in range(3):
            run_name = "norgs%d_rep%d_%dsteps_no_barriers" % (n_orgs, i+1, 2000)
            run = wandb.init(project="organoid_annealing", id=run_name, config=config) # mode="disabled"
            run.config.update({"n_total_orgs": n_orgs}, allow_val_change=True)

            # image generation
            im, centroids = generate_image()
            wandb.log({"seq_gen_array": wandb.Image(im.astype(np.uint8)),
                    "seq_gen_centroids": wandb.Table(data = pd.DataFrame(centroids, columns=["x","y"]))})

            # sim annealing
            new_im, new_centroids, log = sim_anneal(im, centroids.tolist())
            wandb.log({"anneal_array": wandb.Image(new_im.astype(np.uint8)),
                    "anneal_centroids": wandb.Table(data = pd.DataFrame(new_centroids, columns=["x","y"]))})

            # save larger files manually in run
            pickle.dump(log, open(os.path.join(run.dir,"anneal_log.txt"), "wb"))
            run.finish()