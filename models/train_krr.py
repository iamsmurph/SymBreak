from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle

# scale data and fit KNN on all experiments
combined_df = pd.read_csv("data/combined_experiments_df.csv")
feats_df = combined_df[['grad200','density700']]
scaler = MinMaxScaler()
X = scaler.fit_transform(feats_df.values)
y = combined_df.cdx2Dipole.values
model = KernelRidge(kernel="rbf").fit(X, y)

# save the model
pickle.dump(model, open('models/krr_model.pkl', 'wb'))

# save the scaler
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))