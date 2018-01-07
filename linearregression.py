import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/panindra/Desktop/Machine Learning/Homework1/linear_regression_test_data.csv')

data_ = df.iloc[:,1:3]
data = df.iloc[:,1:3]

X_PCA_RAW = df.iloc[:,1]
Y_PCA_RAW = df.iloc[:,2]
Y_THEORITICAL = df.iloc[:,3]

def PCA(data):
    mean = np.mean(data, 0)
    standardised = data - mean
    covariance_matrix = np.cov(data, rowvar= False)
    print(covariance_matrix)
    eigen_value,eigen_vector = np.linalg.eig(np.array(covariance_matrix))
    print("Eigen Value")
    print(eigen_value)
    print("EigenVector")
    print(eigen_vector)
    print(eigen_vector.shape)
    
    sorted_eigen_index = eigen_value.argsort()[::-1]
    
    eigen_value_sorted = eigen_value[sorted_eigen_index]
    vector_component = eigen_vector[:,sorted_eigen_index]
    
    #eigen_max_pos = np.argmax(eigen_value_sorted)
    #eigenVec_max_one = (vector_component[:,eigen_max_pos])
    
    print("Principal Components")
    print(vector_component)
    
    pca_scores = np.dot(standardised,vector_component)
    print(pca_scores[:,0])
    
    return pca_scores, eigen_value_sorted, vector_component, covariance_matrix


pca_scores, eigenval, PCs, cov_mat = PCA(data)



#Find total variance:
total_variance_data = sum(cov_mat.diagonal())

#Total variance of PCs
total_variance_pcs  = sum(np.cov(PCs, rowvar = False).diagonal())

#variance btw pcs
pcs_cov = np.cov(PCs, rowvar = False)[1,0]

actual_pca = pca_scores[:,0:2]


#PCA scores is pca output which we have considered only two here
#Plot of PCs
fig = plt.figure()
fig.suptitle('Scores Plot')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.scatter(pca_scores[:,0],pca_scores[:,1], c='R')
ax.scatter(pca_scores[:,0], pca_scores[:,1], c='B')
plt.show()

#Loadings = Acual eigen vectors
fig = plt.figure()
fig.suptitle('eigen vector')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.scatter(PCs[:,0],PCs[:,1], c='R')
ax.scatter(PCs[:,0], PCs[:,1], c='B')
plt.show()

# plot which has PC1 axis, Y vs X raw, Y vs X theoritical, data points
fig = plt.figure()
fig.suptitle('PC1 axis, Y vs X raw, Y vs X theoritical')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(Y_PCA_RAW,X_PCA_RAW, c='R')
ax.plot(Y_THEORITICAL, X_PCA_RAW, c='B')
#ax.plot(Y_PCA_RAW,pca_scores[:,0],c='G')
#ax.scatter(vector_component[0,0],vector_component[1,0], c='G')
ax.plot([0, 5 * PCs[0,0]], [0, 5 * PCs[1,0]],color='green', linewidth=3)
plt.show()

#Question 2
df = pd.read_csv('/Users/panindra/Desktop/Machine Learning/Homework1/linear_regression_test_data.csv')

data = df.iloc[:,1:3]

data_ = df.iloc[:,1:3]
data_ = df.iloc[:,1:3]
data_ = data_.as_matrix()

X = data_[:,0]
X = np.matrix(X).T

Y = data_[:,1]
Y = np.matrix(Y).T

onesarray = np.ones(len(X))

x_bias = np.c_[onesarray,X]

x_Res = np.dot(x_bias.T, x_bias)
print x_Res

y_Res = np.dot(x_bias.T, Y)
print y_Res

from numpy.linalg import inv

 # inverse of the first part of the equation is done using inv function from numpy
A_inverse = inv(x_Res)
print A_inverse

A = np.dot(A_inverse, y_Res)
print A
A[0]

y_result = A[0] + (A[1] * X.T)
y_result = y_result.T

# plot of LInear regression line and data points
fig = plt.figure()
fig.suptitle('X and Y_result (Linear Regression)')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(data_[:,0],data_[:,1])
ax.plot(data_[:,0],y_result, c='R')
plt.show()


# plot which has PC1 axis & Regression LIne
fig = plt.figure()
fig.suptitle('PC1 axis & Regression LIne')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(Y_PCA_RAW,X_PCA_RAW, c='R')
ax.scatter(Y_THEORITICAL, X_PCA_RAW, c='B')
#ax.plot(Y_PCA_RAW,pca_scores[:,0],c='G')
#ax.plot(vector_component[0,0],vector_component[1,0], c='G')
ax.plot([0, 5 * PCs[0,0]], [0, 5 * PCs[1,0]],color='green', linewidth=3)
ax.plot(data_[:,0],y_result, c='R')
plt.show()
#Answer: 
# Yes, they are slightly similar which we can see in the above graph.


#Question 3
from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
diabetes = datasets.load_diabetes()
diabetes.keys()
diabetes_data = pd.DataFrame(diabetes.data)
diabetes_data.shape[0]

diabetes_data.head()

X_ = diabetes_data[2]
Y_ = diabetes.target

lr_data  = np.column_stack((X_,Y_))
lr_data.shape
lr_data = pd.DataFrame(lr_data)
print lr_data

# as asked in question
num_of_test_samples = 20
num_of_train_samples = 422

# code to select the random samples
#testing indexes
samples_to_select = np.int_(lr_data.shape[0] * np.random.rand(20))
samples_to_select.shape # should be 20
#training indexes
train_sample_index = np.setdiff1d(np.arange(0, 442, 1), samples_to_select)
train_sample_index.shape

#testing samples
diabetes_data_testing = lr_data.iloc[samples_to_select, :]
diabetes_data_testing.shape
type(diabetes_data_testing)

#training samples
training = lr_data.iloc[train_sample_index,:]
training.shape
type(training)


x = np.array(training[0])
y = np.array(training[1])

lm = linear_model.LinearRegression()
x = x.reshape(len(x), 1)
lm.fit(x,y)
y_bar = lm.predict(x)

print lm.intercept_
print lm.coef_
print r2_score(y, y_bar)
print mean_squared_error(y, y_bar)
print y_bar

x = np.array(diabetes_data_testing[0])
y = np.array(diabetes_data_testing[1])
x = x.reshape(len(x), 1)
lm.fit(x,y)
y_bar = lm.predict(x)


print lm.intercept_
print lm.coef_
print r2_score(y, y_bar)
print mean_squared_error(y, y_bar)
print y_bar
y_bar = pd.Series(y_bar)

#linear Reg plot
fig = plt.figure()
fig.suptitle('Linear Regression SKLEARN for diabetes dataset')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(diabetes_data_testing[0],diabetes_data_testing[1], c='R')
ax.scatter(diabetes_data_testing[0],y_bar, c='B')
ax.plot(diabetes_data_testing[0],y_bar, c='G')
plt.show()