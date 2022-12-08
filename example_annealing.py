from pipeline.symbreakanneal import SymBreakAnneal

pipeline = SymBreakAnneal(
        data_path = 'data/example_coords.csv',
        save_path = "results/",
        save_plots = True
    )

    # image generation
im, centroids = pipeline.sample_pattern(save_samples = True)

# sim annealing
#new_im, new_centroids, log = pipeline.sim_anneal(im, centroids.tolist())

# save larger files manually in run
#pickle.dump(log, open(os.path.join("results/","anneal_log.txt"), "wb"))