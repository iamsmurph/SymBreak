# Controlling organoid symmetry breaking uncovers an excitable system underlying human axial elongation

## Quick Links
- Paper: Anand et. al. 2020

## Installation and Usage

After cloning, users are recommended to install requirements in a Conda environment. Access to a GPU and CUDA Toolkit 11.2 or higher is required. 

```
conda create --name <env> --file requirements.txt
```

Users may predict dipoles of a particular spatial configuration of organoids according to the code below. The input should be a .csv file without a header containing the centroid coordinates of organoids. 

```
from pipeline.symbreak import SymBreak

pipeline = SymBreak(
    data_path = "data/example_coords.csv",
    save_path = "results/",
    save_plots = True,
)

results_df = pipeline.predict_dipole()
```

Users may also optimize the spatial patterning of organoids by simulated annealing. In this example, coordinates are sampled. The user's own coordinates may also be inputted during class initialization. 
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
