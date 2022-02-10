import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

IMDIM = 120
NTRAIN = 5
PRED_DIST = 75

def dipole(morph, cdx2):
    edged = cv.Canny(morph, 30, 200)

    # Finding Contours
    contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    ############## Constants ##############
    

    def avg_signal_vec(centroid, coords, sig_mat = None, returnNorm = False):
        # find the direction of the average signal
        pixelSumVec = np.zeros(centroid.shape)
        for coord in coords:
            #print(coord, centroid)
            vec = coord - centroid

            if sig_mat is not None:
                signal = sig_mat[coord[0], coord[1]]
                signal = min(signal, 1000)
                pixelSumVec += vec*signal
            else:
                pixelSumVec += vec
        if returnNorm is True:
            return pixelSumVec, np.sqrt(np.sum(pixelSumVec**2))
        else:
            return pixelSumVec

    signalVecs = []
    signalNorms = []
    for org_ix in range(len(contours)):
        # filled contours
        filled_cnt = cv.drawContours(np.zeros_like(morph), contours, org_ix, 255, thickness=cv.FILLED)

        # centroid of contour
        M = cv.moments(contours[org_ix])
        y = int(M["m10"] / M["m00"])
        x = int(M["m01"] / M["m00"])
        centroid = np.array([x,y])
        
        # cdx2 data structures
        pts = np.array(np.where(filled_cnt == 255))
        cdxx, cdyy = pts[0], pts[1]
        cdx2Coords = np.vstack([cdxx, cdyy]).T

        signalVec, signalVecNorm = avg_signal_vec(centroid, cdx2Coords, sig_mat = cdx2, returnNorm = True)
        signalNorms.append((x, y, signalVecNorm))
        signalVecs.append(signalVec)
    
    return signalNorms, signalVecs


if __name__ == "__main__":
    df1 = cv.imread("datasets/round_1/rawData/morphology/round1_01_segmented_removed_aligned.tiff",0)
    df2 = cv.imread("datasets/round_1/rawData/morphology/round1_02_segmented_removed_aligned.tiff",0)
    df3 = cv.imread("datasets/round_1/rawData/morphology/round1_03_segmented_removed_aligned.tiff",0)
    df4 = cv.imread("datasets/round_1/rawData/morphology/round1_04_segmented_removed_aligned.tiff",0)
    df5 = cv.imread("datasets/round_1/rawData/morphology/round1_05_segmented_removed_aligned.tiff",0)
    df6 = cv.imread("datasets/round_1/rawData/morphology/round1_06_segmented_removed_aligned.tiff",0)
    morphs = [255 - df1, 255 - df2, 255 - df3, 255 - df4, 255 - df5, 255 - df6]

    cdx2_1 = cv.imread('datasets/round_1/rawData/bio/round1_01_CDX2.tiff', 0).astype(np.float64)
    cdx2_2 = cv.imread('datasets/round_1/rawData/bio/round1_02_CDX2.tiff', 0).astype(np.float64)
    cdx2_3 = cv.imread('datasets/round_1/rawData/bio/round1_03_CDX2.tiff', 0).astype(np.float64)
    cdx2_4 = cv.imread('datasets/round_1/rawData/bio/round1_04_CDX2.tiff', 0).astype(np.float64)
    cdx2_5 = cv.imread('datasets/round_1/rawData/bio/round1_05_CDX2.tiff', 0).astype(np.float64)
    cdx2_6 = cv.imread('datasets/round_1/rawData/bio/round1_06_CDX2.tiff', 0).astype(np.float64)
    cdx2s = [cdx2_1, cdx2_2, cdx2_3, cdx2_4, cdx2_5, cdx2_6]

    dipoles = []
    for morph, cdx2 in list(zip(morphs, cdx2s)):
        norms, vecs = dipole(morph, cdx2)
        dipoles.append((np.array(norms), np.array(vecs)))