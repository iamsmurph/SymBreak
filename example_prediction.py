from pipeline.symbreak import SymBreak

pipeline = SymBreak(
    data_path = "data/smaller_example_coords.csv",
    save_path = "results/",
    save_plots = True,
)

results_df = pipeline.predict_dipole()

print(results_df)

