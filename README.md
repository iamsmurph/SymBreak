# Controlling organoid symmetry breaking uncovers an excitable system underlying human axial elongation

## Quick Links
- Paper: Anand et. al. 2020

## Usage

After cloning, users can predict dipoles of a particular spatial configuration of organoids according to the code below. Input file should be a headless .csv format containing the centroid coordinates of organoids. 

```
from pipeline.symbreak import SymBreak

pipeline = SymBreak(
    data_path = "data/example_coords.csv",
    save_path = "results/",
    save_plots = True,
)

results_df = pipeline.predict_dipole()
```

Users may also optimize the spatial patterning of organoids by simulated annealing. In this example, coordinates are sampled.
```
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
```

## Contact
Please contact sharad@cgr.harvard.edu with questions.
