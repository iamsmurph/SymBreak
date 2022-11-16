#import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
import pickle
from sklearn.preprocessing import MinMaxScaler
import argparse
import cv2
import time
from imageio import imwrite
from PIL import Image


def get_features(mask, centroids):
        # generate pattern
        print("Generating patterns...", end="")
        im = make_pattern(mask, centroids, fillVal = 255).astype(np.uint8)
        print("Done")
        #im = Image.fromarray(im)
        #yo = data.astronaut()
        #breakpoint()        
        
        print("Gaussian blurring...")
        im_blurs = []
        sigmas = [200, 700]
        for sigma in tqdm(sigmas):
            #im = np.asarray(im)
            #start = time.time()
            #blur1 = cv2.GaussianBlur(im, (0, 0), sigma, sigma)
            #end = time.time()
            #print(end - start)
            #cv2.imwrite(f"cv2_test_im_{sigma}.jpg", blur1)   
            blur = ndimage.gaussian_filter(im, sigma=sigma) #, mode = 'constant'
            im_blurs.append(blur)
        #breakpoint()

        print("Computing features...", end="")
        grad_rho200 = compute_feats(im, im_blurs[0], centroids, rho = False)
        rho700 = compute_feats(im, im_blurs[1], centroids, grad_rho = False)
        feats = np.array(list(zip(rho700, grad_rho200)))
        print("Done")
        return feats
        
def compute_feats(im, im_blur, centroids, rho = True, grad_rho = True):
    feats = []
    for x, y in centroids:
        
        bool_mat = circle(x,y, im, org_rad)

        dfeats = im_blur[bool_mat]
        if rho and not grad_rho:
            density = np.mean(dfeats)
            feats.append(density)
        if not rho and grad_rho:
            grad, _ = max_gradient(x,y, im_blur)
            feats.append(grad)
        if rho and grad_rho:
            feats.append((density, grad))
    return feats
            

def max_gradient(x,y, im_blur):
    # get pixel intensities in extremes
    xmax = im_blur[x + org_rad - 1, y]
    xmin = im_blur[x - org_rad, y]
    ymax = im_blur[x, y + org_rad - 1]
    ymin = im_blur[x, y - org_rad]

    xdiffnorm  = ((int(xmax) - int(xmin)) / org_rad)
    ydiffnorm = ((int(ymax) - int(ymin)) / org_rad)

    # gradient magnitude and vector
    grad = np.array(np.sqrt(xdiffnorm**2 + ydiffnorm**2))
    gradVec = np.array((xdiffnorm, ydiffnorm))
                
    return grad, gradVec

def make_pattern(mask, centroids, fillVal = 255):
    for x,y in centroids:
        dim = len(mask)
        xx, yy = np.mgrid[:org_rad*2, :org_rad*2]
        zz = (xx - org_rad) ** 2 + (yy - org_rad) ** 2
        circle = zz < org_rad ** 2
        bool_mat = np.pad(circle, ((x-org_rad, dim-x-org_rad),(y-org_rad, dim-y-org_rad)))
        mask[bool_mat] = fillVal
    return mask

def make_plot(mask, centroids, preds):
    scaler = MinMaxScaler(feature_range=(0, 255))
    preds_norm = scaler.fit_transform(preds)
    for (x,y), pred in zip(centroids, preds_norm):
        dim = len(mask)
        xx, yy = np.mgrid[:org_rad*2, :org_rad*2]
        zz = (xx - org_rad) ** 2 + (yy - org_rad) ** 2
        circle = zz < org_rad ** 2
        bool_mat = np.pad(circle, ((x-org_rad, dim-x-org_rad),(y-org_rad, dim-y-org_rad)))
        mask[bool_mat] = pred
    return mask

def circle(x,y, orgIm, radius, fill = False, fillVal = 255):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    parser.add_argument("--save_csv", action='store_true')
    parser.add_argument("--save_img", action='store_true')
    args = parser.parse_args()

    centroids = pd.read_csv(args.f).values.astype(int)
    
    org_rad = 75
    pad = 1000
    model_path = "models/krr_model.checkpoint"

    cmin = centroids.min()
    if cmin != pad:
        diff = pad - cmin
        centroids += diff
    
    size = int(centroids.max() + pad)
    mask = np.zeros((size, size))
    feats = get_features(mask, centroids)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(feats)

    loaded_model = pickle.load(open(model_path, 'rb'))
    preds = loaded_model.predict(X)
    preds = preds.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    preds_norm = scaler.fit_transform(preds)

    if args.save_csv:
        arr = np.hstack([centroids, feats, preds_norm])
        cols = ["cx", "cy","density_700","grad_200","pred"]
        res_df = pd.DataFrame(arr, columns = cols)
        res_df.to_csv("res_df.csv", index=False)
    
    if args.save_img:
        mask = np.zeros((size, size))
        res_plot = make_plot(mask, centroids, preds_norm)
        fig, ax = plt.subplots()
        im = ax.imshow(res_plot)
        plt.colorbar(im)
        plt.title("Dipole Prediction Plot")
        fn = 'result_plot.png'
        plt.savefig(fn, format='png')