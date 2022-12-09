from pipeline.symbreakanneal import SymBreakAnneal
import os
import pickle

pipeline = SymBreakAnneal(
        data_path = 'data/example_coords.csv',
        save_path = "results/",
        save_plots = True,
        n_total_orgs = 6
    )

    # image generation
im, centroids = pipeline.sample_pattern()

# sim annealing
new_im, new_centroids, log = pipeline.sim_anneal(im, centroids.tolist())

assert(len(centroids) == len(new_centroids))

# save larger files manually in run
#pickle.dump(log, open(os.path.join("results/","anneal_log.txt"), "wb"))