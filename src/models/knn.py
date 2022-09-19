from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle

# scale data and fit KNN on all experiments
combined_df = pd.read_csv("circle_combined_df.csv")
feats_df = combined_df[['grad200','density700']]
scaler = MinMaxScaler()
X = scaler.fit_transform(feats_df.values)
y = combined_df.cdx2Dipole.values
model = KernelRidge(kernel="rbf").fit(X, y)

# save the model to disk
filename = 'knn_model.checkpoint'
pickle.dump(model, open(filename, 'wb'))