import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp
from cupyx.scipy import ndimage
import pickle


def generate_window(seed, p):
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


def extract_features(image, sigma, centroids):

    im_blur = ndimage.gaussian_filter(cp.array(image), sigma=sigma, mode='constant',cval=0)
    im_blur_norm=im_blur*sigma*cp.sqrt(np.pi)
    
    #plt.imshow(im_blur_norm.get())
    #name = "gaussian_" + str(sigma)+ "_" + str(seed) + ".jpeg"
    #plt.axis("off")
    #plt.show()
    #plt.savefig(os.path.join(save_dir, name), bbox_inches='tight', transparent=True, pad_inches=0, dpi=200)

    im_sx = ndimage.sobel(im_blur_norm, axis=1, mode='reflect')
    im_sy = ndimage.sobel(im_blur_norm, axis=0, mode='reflect')
    im_sobel=np.hypot(im_sx, im_sy)

    #plt.imshow(im_sobel.get())
    #plt.axis("off")
    #name = "sobel_" + str(sigma)+ "_" + str(seed) + ".jpeg"
    #plt.show()
    #plt.savefig(os.path.join(save_dir, name), bbox_inches='tight', transparent=True, pad_inches=0, dpi=200)

    feats = []
    
    for centroid in centroids:
        x, y = centroid[0], centroid[1]
        density = cp.mean(im_blur_norm[x-75: x+75, y-75: y+75])
        grad = cp.mean(im_sobel[x-75: x+75, y-75: y+75])
        feats.append([density.get(), grad.get()])

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    feats = np.array(feats)
    return feats

pattern_dim = 40
p = 1/4
pad = 200

np.round(1/6,3)

all_feat_names = []
ps = [1/4, 1/6, 1/8, 1/12, 1/16, 1/24, 1/32]                    
for p in ps:
    seeds = [1,2,4,5,9,10]
    for seed in seeds:
        im, centers = generate_window(seed, p)
        for sigma in list(range(200, 1200, 200)):
            name = str(np.round(p,3)) +"_"+str(seed)+"_"+str(sigma)
            all_feat_names.append(name)

pickle.dump(all_feat_names, open("all_feat_names.txt", "wb"))    

all_feats = []
ps = [1/4, 1/6, 1/8, 1/12, 1/16, 1/24, 1/32]                    
for p in tqdm(ps):
    seeds = [1,2,4,5,9,10]
    for seed in seeds:
        im, centers = generate_window(seed, p)
        for sigma in list(range(200, 1200, 200)):
            all_feats.append(extract_features(im, sigma, centers))


pickle.dump(all_feats, open("all_feats.txt", "wb"))