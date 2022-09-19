import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
import pickle
from sklearn.preprocessing import MinMaxScaler
import io

#import matplotlib.image as image

st.title("Machine learning directed organoid morphogenesis")

def get_features(mask, centroids):
        # generate pattern
        st.text("Generating patterns...    ")
        im = make_pattern(mask, centroids, fillVal = 255)
        
        #print("Generating pattern...")
        #for x,y in tqdm(centroids):
        #    dim = len(mask)
        #    xx, yy = np.mgrid[:org_rad*2, :org_rad*2]
        #    zz = (xx - org_rad) ** 2 + (yy - org_rad) ** 2
        #    circle = zz < org_rad ** 2
        #    bool_mat = np.pad(circle, ((x-org_rad, dim-x-org_rad),(y-org_rad, dim-y-org_rad)))
        #    mask[bool_mat] = 255
        #im = mask
        st.text("Done.")
        
        st.text("Gaussian blurring...")
        im_blurs = []
        for sigma in tqdm([200, 700]):
            blur = ndimage.gaussian_filter(im, sigma=sigma, mode = 'constant')
            im_blurs.append(blur)
        st.text("Done.")

        st.text("Computing features...")
        grad_rho200 = compute_feats(im, im_blurs[0], centroids, rho = False)
        rho700 = compute_feats(im, im_blurs[1], centroids, grad_rho = False)
        st.text("Done.")

        feats = np.array(list(zip(rho700, grad_rho200)))
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

    xdiffnorm  = ((xmax - xmin) / org_rad)
    ydiffnorm = ((ymax - ymin) / org_rad)

    # gradient magnitude and vector
    grad = np.array(np.sqrt(xdiffnorm**2 + ydiffnorm**2))
    gradVec = np.array((xdiffnorm, ydiffnorm))
                
    return grad, gradVec

def make_pattern(mask, centroids, fillVal = 255):
    n = len(centroids)
    for i in tqdm(range(n)):
        x,y = centroids[i]
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

size = st.number_input('Input mask size:')
uploaded_file = st.file_uploader("Upload organoid coordinates (.csv format):")
size = int(size)

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    centroids = pd.read_csv(uploaded_file).values[:20].astype(int)
    
    #st.write(centroids, width = 100)
    #st.write("yo ", size)
    
    org_rad = 75
    model_path = "knn_model.checkpoint"

    #centroids = pd.read_csv("test_coords.csv").values[:20].astype(int)
    centroids = centroids // 4
    mask = np.zeros((size, size))
    feats = get_features(mask,centroids)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(feats)

    loaded_model = pickle.load(open(model_path, 'rb'))
    preds = loaded_model.predict(X)
    preds = preds.reshape(-1, 1)
    res_arr = np.hstack([centroids, feats, preds])
    cols = ["cx", "cy","density_700","grad_200","pred"]
    res_df = pd.DataFrame(res_arr, columns = cols)
    @st.cache
    
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    st.subheader("Prediction Table")
    csv = convert_df(res_df)
    st.write(res_df)
    st.download_button(
        "Download",
        csv,
        "result_table.csv",
        "text/csv",
        key='download-csv'
    )

    #res_df.to_csv("result_df.csv", index = False)
    st.subheader("Prediction Plot")
    mask = np.zeros((size, size))
    scaler = MinMaxScaler(feature_range=(0,1))
    preds_norm = scaler.fit_transform(preds)
    res_plot = make_plot(mask, centroids, preds_norm)
    #image.imsave('result.png', hi)
    fig, ax = plt.subplots()
    im = ax.imshow(res_plot)
    plt.colorbar(im)
    plt.title("Dipole Prediction Plot")
    
    fn = 'result_plot.png'
    img = io.BytesIO()
    plt.savefig(img, format='png')

    st.pyplot(fig)
    
    btn = st.download_button(
    label="Download",
    data=img,
    file_name=fn,
    mime="image/png"
    )