from organoid import preprocessing
import numpy as np
from collections import Counter

# experiment id
expNum = 12

#data dir
dir = "random12_design"

# loading centroid arrays
initPattern = np.load("datasets/" + dir + "/designRandom" + str(expNum) + ".npy")
finalPattern = np.load("datasets/" + dir + "/coordsRewardRandom" + str(expNum) + ".npy")

# instantiate preprocessing class with save dir path
p = preprocessing.Alignment(dir)

p.matching(initPattern, finalPattern, searchLen = 1000, normScalar = 200, validation = True)