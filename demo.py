import numpy as np
import pandas as pd
# eig_vals = np.array([1, 2, 3, 4])
# eig_vecs = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

# asd = np.abs(eig_vals)
matrix = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
eig_vals, eig_vecs = np.linalg.eig(matrix)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
w = pd.DataFrame.merge((eig_pairs[i][1].reshape(4, 1)) for i in range(len(eig_vals)))

a = np.array([1,2,3,4,5,6,7,8,9,10])
b = a.reshape(10,1)
print(a)
print(b)