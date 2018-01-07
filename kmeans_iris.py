import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data
Frame=pd.DataFrame(data = X, columns = ["x", "y","z","w"])
df = Frame
k = 3

centroids = {
    i+1: [df['x'][i], df['y'][i], df['z'][i], df['w'][i]]
    for i in range(k)
}
print("Initial centriods")
print(np.asmatrix(centroids))
    
colmap = {1: 'r', 2: 'g', 3: 'b',4:'y'}

def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2 
                + (df['z'] - centroids[i][2]) ** 2
                + (df['w'] - centroids[i][3]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df
df = assignment(df, centroids)
print(df.head())

import copy

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        centroids[i][2] = np.mean(df[df['closest'] == i]['z'])
        centroids[i][3] = np.mean(df[df['closest'] == i]['w'])
    return k

centroids = update(centroids)

for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    old_z = old_centroids[i][2]
    old_w = old_centroids[i][3]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    dz = (centroids[i][2] - old_centroids[i][2]) * 0.75
    dw = (centroids[i][3] - old_centroids[i][3]) * 0.75


## Repeat Assigment Stage

df = assignment(df, centroids)
df.shape

count = 0
# Continue until all assigned categories don't change any more
while True:
    count = count + 1
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'])
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(-0.5,15.0)
plt.ylim(-0.5,15.0)
plt.show()

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['z'], df['w'], color=df['color'])
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(-0.5,15.0)
plt.ylim(-0.5,15.0)
plt.show()
df[df.columns[:4]]

#from sklearn.metrics import silhouette_samples, silhouette_score
#silhouette_avg = silhouette_score(df[df.columns[:4]], df['closest'])
#print("For n_clusters =",
  #        "The average silhouette_score is :", silhouette_avg)