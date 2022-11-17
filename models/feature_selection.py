
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SequentialFeatureSelector
import seaborn as sns
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

df = pd.read_csv("circle_combined_df_big.csv")

X = df.iloc[:, 5:-1]
y = df.iloc[:, -1]

estimator = KernelRidge(kernel="rbf")
results = []

for n in tqdm(range(1, 4)):
    sfs = SequentialFeatureSelector(estimator, n_features_to_select=n, scoring="neg_root_mean_squared_error")
    sfs.fit(X, y)
    
    select_df = X.loc[:, sfs.get_support()]
    cols = select_df.columns
    
    scores = cross_val_score(estimator, select_df, y, cv=5, scoring="neg_root_mean_squared_error")
    result = scores.mean()
    print(result)
    print(np.std(y))

    results.append((cols, result))


with open("results_feature_selection", "wb") as fp:
    pickle.dump(results, fp)