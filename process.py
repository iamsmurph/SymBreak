from utils import preprocessing
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pickle
import pandas as pd

savedir = "datasets/round_1/matched"
rawdir = "datasets/round_1/rawData/"

# Remove
design_f = "round0_4_seed10_p0.25.txt"
#design_f = "round0_5_seed2_p0.25.txt"
#design_f = "round0_6_seed9_p0.25.txt"
#design_f = "round0_1_seed10_p0.125.txt"
#design_f = "round0_2_seed2_p0.167.txt"
#design_f = "round0_3_seed1_p0.125.txt"

final_f = 'exported_metrics_round1_random01_n4.csv'
#final_f = 'exported_metrics_round1_random02_n4.csv'
#final_f = 'exported_metrics_round1_random03_n4.csv'
#final_f = 'exported_metrics_round1_random04_n4.csv'
#final_f = 'exported_metrics_round1_random05_n4.csv'
#final_f = 'exported_metrics_round1_random06_n4.csv'

# loading design and final centroids
designPattern = np.genfromtxt(os.path.join(rawdir, design_f), delimiter=',')
finalPattern = pd.read_csv(os.path.join(rawdir, final_f))
finalPattern = finalPattern.values[:, 3:5]

# instantiate preprocessing class with save dir path
p = preprocessing.Alignment(rawdir)

new_designPattern, new_finalPattern = p.removal(designPattern, finalPattern)
assert(len(new_designPattern) == len(new_finalPattern))

removeCheckFig = plt.figure(1)
plt.scatter(new_designPattern[:,0], new_designPattern[:,1], c = "blue")
plt.scatter(new_finalPattern[:,0], new_finalPattern[:,1], c="orange")
plt.title("Removed: Design and Final Coords")
plt.show()

new = "removed_"
np.save(open(new+os.path.splitext(design_f)[0] + ".npy", "wb"), new_designPattern)
np.save(open(new+os.path.splitext(final_f)[0] + ".npy", "wb"), new_finalPattern)

p = preprocessing.Alignment(savedir)
p.matching(new_designPattern, new_finalPattern, validation=True)