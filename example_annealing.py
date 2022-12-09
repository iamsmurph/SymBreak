from pipeline.symbreakanneal import SymBreakAnneal

pipeline = SymBreakAnneal(
        save_path = "results/",
        save_plots = True,
        n_total_orgs = 6
    )

# sample coordinates
centroids = pipeline.sample_pattern()

# sim annealing
optim_centroids = pipeline.sim_anneal(centroids)