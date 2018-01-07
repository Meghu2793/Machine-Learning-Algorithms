import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("SCLC_study_output_filtered_2.csv",delimiter=",")
data = data[1:,1:]
Frame=pd.DataFrame(data = data, columns = ["x","y","z","w","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o"])
df = Frame
k = 2

centroids = {
    i+1: [df['x'][i], df['y'][i], df['z'][i],df['w'][i], df['a'][i], df['b'][i],df['c'][i], df['d'][i], df['e'][i],
          df['f'][i], df['g'][i], df['h'][i],df['i'][i], df['j'][i], df['k'][i],df['l'][i], df['m'][i], df['n'][i],df['o'][i]]
    for i in range(k)
}
cluster_colors = {1: 'r', 2: 'g', 3: 'b'}
 """   
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'],df['z'], color='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color = cluster_colors[i])
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.show()
"""
def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2 
                + (df['z'] - centroids[i][2]) ** 2
                + (df['w'] - centroids[i][3]) ** 2 
                + (df['a'] - centroids[i][4]) ** 2
                + (df['b'] - centroids[i][5]) ** 2 
                + (df['c'] - centroids[i][6]) ** 2
                + (df['d'] - centroids[i][7]) ** 2 
                + (df['e'] - centroids[i][8]) ** 2
                + (df['f'] - centroids[i][9]) ** 2 
                + (df['g'] - centroids[i][10]) ** 2
                + (df['h'] - centroids[i][11]) ** 2 
                + (df['i'] - centroids[i][12]) ** 2
                + (df['j'] - centroids[i][13]) ** 2 
                + (df['k'] - centroids[i][14]) ** 2
                + (df['l'] - centroids[i][15]) ** 2 
                + (df['m'] - centroids[i][16]) ** 2
                + (df['n'] - centroids[i][17]) ** 2
                + (df['o'] - centroids[i][18]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: cluster_colors[x])
    return df
df = assignment(df, centroids)
print(df.head())
"""
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=cluster_colors[i])
plt.xlim(-1,5)
plt.ylim(-1, 5)
plt.show()
"""
import copy

prev_centriods = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        centroids[i][2] = np.mean(df[df['closest'] == i]['z'])
        centroids[i][3] = np.mean(df[df['closest'] == i]['w'])
        centroids[i][4] = np.mean(df[df['closest'] == i]['a'])
        centroids[i][5] = np.mean(df[df['closest'] == i]['b'])
        centroids[i][6] = np.mean(df[df['closest'] == i]['c'])
        centroids[i][7] = np.mean(df[df['closest'] == i]['d'])
        centroids[i][8] = np.mean(df[df['closest'] == i]['e'])
        centroids[i][9] = np.mean(df[df['closest'] == i]['f'])
        centroids[i][10] = np.mean(df[df['closest'] == i]['g'])
        centroids[i][11] = np.mean(df[df['closest'] == i]['h'])
        centroids[i][12] = np.mean(df[df['closest'] == i]['i'])
        centroids[i][13] = np.mean(df[df['closest'] == i]['j'])
        centroids[i][14] = np.mean(df[df['closest'] == i]['k'])
        centroids[i][15] = np.mean(df[df['closest'] == i]['l'])
        centroids[i][16] = np.mean(df[df['closest'] == i]['m'])
        centroids[i][17] = np.mean(df[df['closest'] == i]['n'])
        centroids[i][18] = np.mean(df[df['closest'] == i]['o'])
    return k

centroids = update(centroids)
"""  
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['x'], df['y'],df['z'], color=df['color'], edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=cluster_colors[i])
plt.xlim(-1,5)
plt.ylim(-1, 5)
"""
for i in prev_centriods.keys():
    old_x = prev_centriods[i][0]
    old_y = prev_centriods[i][1]
    old_z = prev_centriods[i][2]
    old_w = prev_centriods[i][3]
    old_a = prev_centriods[i][4]
    old_b = prev_centriods[i][5]
    old_c = prev_centriods[i][6]
    old_d = prev_centriods[i][7]
    old_e = prev_centriods[i][8]
    old_f = prev_centriods[i][9]
    old_g = prev_centriods[i][10]
    old_h = prev_centriods[i][11]
    old_i = prev_centriods[i][12]
    old_j = prev_centriods[i][13]
    old_k = prev_centriods[i][14]
    old_l = prev_centriods[i][15]
    old_m = prev_centriods[i][16]
    old_n = prev_centriods[i][17]
    old_o = prev_centriods[i][18]
    dx = (centroids[i][0] - prev_centriods[i][0]) * 0.75
    dy = (centroids[i][1] - prev_centriods[i][1]) * 0.75
    dz = (centroids[i][2] - prev_centriods[i][2]) * 0.75
    dw = (centroids[i][0] - prev_centriods[i][3]) * 0.75
    da = (centroids[i][1] - prev_centriods[i][4]) * 0.75
    db = (centroids[i][2] - prev_centriods[i][5]) * 0.75
    dc = (centroids[i][0] - prev_centriods[i][6]) * 0.75
    dd = (centroids[i][1] - prev_centriods[i][7]) * 0.75
    de = (centroids[i][2] - prev_centriods[i][8]) * 0.75
    df = (centroids[i][0] - prev_centriods[i][9]) * 0.75
    dg = (centroids[i][1] - prev_centriods[i][10]) * 0.75
    dh = (centroids[i][2] - prev_centriods[i][11]) * 0.75
    di = (centroids[i][0] - prev_centriods[i][12]) * 0.75
    dj = (centroids[i][1] - prev_centriods[i][13]) * 0.75
    dk = (centroids[i][2] - prev_centriods[i][14]) * 0.75
    dl = (centroids[i][1] - prev_centriods[i][15]) * 0.75
    dm = (centroids[i][2] - prev_centriods[i][16]) * 0.75
    dn = (centroids[i][0] - prev_centriods[i][17]) * 0.75
    do = (centroids[i][1] - prev_centriods[i][18]) * 0.75
#ax.arrow(old_x, old_y,old_z, dx, dy,dz, head_width=2, head_length=3, fc=cluster_colors[i], ec=cluster_colors[i])
plt.show()

## Repeat Assigment Stage

df = assignment(df, centroids)

# Plot results
"""
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'],df['z'], color=df['color'], edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=cluster_colors[i])
plt.xlim(-1,5)
plt.ylim(-1, 5)
plt.show()
"""
# Continue until all assigned categories don't change any more
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break
"""
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'],df['z'], color=df['color'])
for i in centroids.keys():
    plt.scatter(*centroids[i], color=cluster_colors[i])
plt.xlim(-0.5,2.0)
plt.ylim(-0.5,6.0)
plt.show()
"""