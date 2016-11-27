import pandas as pd
import numpy as np

wines = pd.read_pickle('both_wine.pkl')
wines_matrix = wines.as_matrix()

def col_mean_vector(d):
	means = np.array
	for col in d.T:
		np.append(means,np.mean(col))
	return means

wines_mean_vector = np.array([col_mean_vector(wines_matrix)])



print('Mean Vector:\n',wines_mean_vector)