import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import argparse
import cv2
import os
import matplotlib.pyplot as plt

def extract_features(image, sigma, centroids, org_rad):

    im_blur = ndimage.gaussian_filter(cp.array(image), sigma=sigma, mode='constant',cval=0)
    im_blur_norm=im_blur*sigma*cp.sqrt(np.pi)

    plt.imshow(im_blur_norm.get())
    plt.savefig("yerr" + str(np.random.choice(2000)))

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
    print(feats.shape)
    return feats

if __name__=="__main__":
    matchPath = "datasets/round_1/round_1_match_data/"
    phenoPath = "datasets/round_1/round_1_phenotypes/"

    match_fs = os.listdir(matchPath)
    pheno_fs = os.listdir(phenoPath)
    match_fs = sorted(match_fs, key=lambda x: int(x[-5]))
    pheno_fs = sorted(pheno_fs, key=lambda x: int(x[-6]))
    print(match_fs)
    print(pheno_fs)
    all_feats = []
    sigmas = [200, 400, 600, 800, 1000]

    big_df = []
    #assert(True == False)
    for match_f, pheno_f in zip(match_fs, pheno_fs):
        feats_subdf = []
        for sigma in sigmas:
            print("Extracting features for "+match_f+" and "+pheno_f+" for sigma "+str(sigma))
            img = cv2.imread(os.path.join(phenoPath, pheno_f), 0).astype(np.float64)
            img_resize = cv2.resize(img, (8400,8400), interpolation = cv2.INTER_AREA)
            centroids = np.load(os.path.join(matchPath, match_f))
            centroids = centroids[:, -2:]
            feats = extract_features(img_resize, sigma, centroids, 75)
            feats_subdf.append(feats)
            
        name = np.repeat(int(match_f[-5]), len(feats_subdf[0])).reshape(-1,1)
        subdf = np.hstack(feats_subdf)
        info = np.append(centroids, name, axis = 1)
        df = np.append(subdf, info, axis = 1)
        big_df.append(df)
        

    np.save('round_1_features.npy', np.vstack(big_df))