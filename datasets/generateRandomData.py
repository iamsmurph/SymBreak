import numpy as np
from tqdm import tqdm
import pickle
import os

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

def simulate(image, centers, weights = "uniform"):
    if weights == "uniform":
        w = [1/len(centers)]*len(centers)
    test_img = image.copy()
    test_centers = centers.copy()
    
    nx, ny, ix = random_move(test_centers, w)
    old_x, old_y = test_centers[ix]
    test_img[old_x-75: old_x+75, old_y-75:old_y+75] = 50
    test_img[nx-75:nx+75, ny-75:ny+75] = 255
    test_centers[ix] = (nx, ny)
    
    return test_img, test_centers

def random_move(centers, weights):
    found_valid_move = False
    newx, newy, random_index = 0, 0, 0
    
    while not found_valid_move:
        cixs = list(range(len(centers)))
        
        random_index = np.random.choice(cixs, 1, p=weights)[0]

        angle = np.pi * np.random.uniform(0, 2)
        length = perturb_len
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
    if (coord < 200) or (coord > imSize- 200):
        return False
    else:
        return True

def random_perturb(im, centroids, niter):
    for _ in range(niter):
        im, centroids = simulate(im, centroids)
    return im, centroids


if __name__ == '__main__':
    # configs
    pattern_dim = 40
    p = 1/4
    pad = 200
    perturb_len = 50
    perturb_iter = 500
    imSize = 8400
    curr_dir = os.path.abspath(os.getcwd())
    save_dir = os.path.join(curr_dir, "datasets")

    hi = []
    pickle.dump(hi, open(os.path.join(save_dir, "test.txt"), "wb"))

    # generate images for different probabilities and seeds
    unlabeled = []
    ps = np.round(np.linspace(1/16, 8/16, 16, 2), 3)
    for p in ps:
        seeds = list(range(0, 10))
        for seed in seeds:
            im, centers = generate_window(seed, p)
            unlabeled.append((im, centers))

    # perturb generated images
    for i, pattern in tqdm(enumerate(unlabeled)):
        im, centroids = pattern[0], pattern[1]
        im, centroids = random_perturb(im, centroids, perturb_iter)
        unlabeled[i] = (im, centroids)

    # save
    pickle.dump(unlabeled, open(os.path.join(save_dir, "unlabeled_data.txt"), "wb"))