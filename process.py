from organoid import preprocessing
import numpy as np
import pandas as pd

# loading centroid arrays
#centroids = np.load("centroids.npz")

#_, coordsPh = centroids.files
#pattEnd = centroids[coordsPh]
pattEnd = np.load("heitorCentroids.npy")
pattInit = np.load("centroidsInitCorrected.npy")

# instantiate preprocessing class with save directory
p = preprocessing.Alignment('scripting/')
# get groups
coordsInit, colInit, coordsPheno, colPheno = p.grouping(pattInit, pattEnd)
# correct groups interactively
print(coordsInit.shape)
print(len(colInit))
print(coordsPheno.shape)

newColPheno = p.coloring(coordsInit, colInit, coordsPheno, colPheno)
print(len(newColPheno))

# save corrected coordinates and col#ors
#resArrs = np.array([coordsInit[:,0], coordsInit[:,1], colInit, coordsPheno[:,0], coordsPheno[:,1], newColPheno])
#cNames = ['xInit', 'yInit', 'colorInit', 'xPheno', 'yPheno', 'colorPheno']

#df = pd.DataFrame(data = resArrs, columns = cNames)
#df.to_csv('matchedDF.csv', index=False)