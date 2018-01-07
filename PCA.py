import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

reader = pd.read_csv('/Users/panindra/Desktop/PCA/dataset.csv')

df = pd.DataFrame(reader)
print(df)

dfa = df
type(dfa)

count = 0
for columns in df.columns:
    count = count + 1

mean_array = dfa.mean(axis=0)
type(mean_array)
mean_cols = np.asarray(mean_array)
type(mean_cols)

columnMeanAll = np.tile(mean_cols,(dfa.shape[0],1))
type(columnMeanAll)
len(columnMeanAll)
xMeanCentered = dfa - columnMeanAll

np.dataframe(xMeanCentered)
type(xMeanCentered)

df = xMeanCentered



rows, cols = df.shape
print("Rows=", rows, " Col=", cols)

type(df)

samples = np.array(df.loc[:,:])
type(samples)

sample_matrix = np.asmatrix(df.loc[:,:])
type(sample_matrix)

samples = samples.T

print(samples)

mean_vector1 = []
for i in range(len(samples)):
    mean_vector1 = np.append(mean_vector1,np.mean(samples[i,:]))

print('Mean Vector:\n', mean_vector1)

covariance_mat = np.cov(samples)
cov1 = np.cov(samples[0]) + np.cov(samples[1]) + np.cov(samples[2])


# eigenvectors and eigenvalues for the from the covariance matrix
eigen_val, eigen_vec = np.linalg.eig(covariance_mat)
print("eig_val_cov\n",eigen_val)
print("eig_vec_cov\n",eigen_vec)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_val[i]), eigen_vec[:,i]) for i in range(len(eigen_val))]
print(eigen_pairs)
type(eigen_pairs)
eigen_pairs[0][1]

# Sort the (eigenvalue, eigenvector) tuples from high to low
sorted_eigen_valVectpairs = eigen_pairs.sort(key=lambda y: y[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eigen_pairs:
    print(i[0])
    
##Get the k value for reduction 
sum_eigVal = np.sum(eigen_val)
arry = []
for i in eigen_val:
    if(i/sum_eigVal > 0.02):
        arry.append(i)
type(arry)
dim = len(arry)

#TODO
matrix_res = []
for i in range(0,dim):
    matrix_res.append(np.hstack(eigen_pairs[i][1].reshape(cols,1)))

print('Matrix W:\n', matrix_res)
type(matrix_res)
matrix1 = np.asmatrix(matrix_res).T

pca_reduced = matrix1.T.dot(samples)

print(pca_reduced)
pca_reduced.shape
pca_reduced = pca_reduced.T
pca_reduced.shape
pca_reduced = pd.DataFrame(pca_reduced)
type(pca_reduced)
fig = plt.figure()
fig.add_subplot(1,1,1)
type(samples)
plt.scatter(pca_reduced.loc[:,0],pca_reduced.loc[:,1])
plt.show()
np.cov(pca_reduced)
cov2 = np.cov(pca_reduced[0]) + np.cov(pca_reduced[1])

#using PCA library
from matplotlib.mlab import PCA

df = pd.DataFrame(reader)
all_samples = np.array(df.loc[:,:])
result = PCA(all_samples)
result.Y
result.fracs

fig = plt.figure()
fig.add_subplot(1,1,1)
plt.scatter(result.Y[:,0],result.Y[:,1])
plt.show()
cov1
cov2
np.cov(result.Y[:,0]) + np.cov(result.Y[:,1])