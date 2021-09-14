from organoid import preprocessing
import numpy as np
from collections import Counter

# loading centroid arrays
initPattern = np.load("datasets/random3/initRandom3.npy")
finalPattern = np.load("datasets/random3/coordsRewardRandom3.npy")

# experiment name
dir = "random3"
# instantiate preprocessing class with save dir path
p = preprocessing.Alignment(dir)

p.matching(initPattern, finalPattern, searchLen = 1000, normScalar = 200, validation = True)