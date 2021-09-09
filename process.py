from organoid import preprocessing
import numpy as np
import pandas as pd
import os
import sys
from collections import Counter

#args = sys.argv
#init, final, saveDir, expName = args[0], args[1], args[2], args[3]

# loading centroid arrays
initPattern = np.load("initRandom7.npy")
finalPattern = np.load("coordReward.npy")

# experiment name
dir = "random7"
# instantiate preprocessing class with save dir path
p = preprocessing.Alignment(dir)
# get groups
#coordsInit, colInit, coordsPheno, colPheno = p.grouping(initPattern, finalPattern) # add saveDir, expName 

# correct groups interactively
#newColPheno = p.coloring(coordsInit, colInit, coordsPheno, colPheno)

# save corrected coordinates and colors in dataframe
#resArrs = np.array([coordsInit[:,0], coordsInit[:,1], colInit, coordsPheno[:,0], coordsPheno[:,1], newColPheno])
#cNames = ['xInit', 'yInit', 'colorInit', 'xPheno', 'yPheno', 'colorPheno']
#df = pd.DataFrame(data = resArrs.T, columns = cNames)

# final processing


# SAVE MATCHED DF
#expName = "random7"
#df.to_csv(os.path.join(saveDir, 'matchedDF_' + expName), index=False)

p.matching(initPattern, finalPattern, dir, searchLen = 1000, normScalar = 200, validation = True) 