import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.cluster.vq import kmeans, vq, whiten





random.seed(777)
x_coord = random.sample(range(0,100), 100)
x_coord = [float(x_coord[i]) for i in range(len(x_coord))]
y_coord = random.sample(range(0,100), 100)
y_coord = [float(y_coord[i]) for i in range(len(y_coord))]

#Normalizacion de los datos
x_scaled = whiten(x_coord)
y_scaled = whiten(y_coord)

df = pd.DataFrame({'X' : x_scaled,
                   'Y' : y_scaled})


plt.figure()

sns.scatterplot(data = df, x = 'X', y = 'Y')

#Hierarchical clustering
Z = linkage(df, 'ward')

dn = dendrogram(Z)

df['cluster_labels'] = fcluster(Z, 3, criterion = 'maxclust')

plt.figure()

sns.scatterplot(data = df, x = 'X', y = 'Y', hue = 'cluster_labels', palette = ['#133468', '#E4AC15', '#E415C5'])
plt.title('Hierarchical Clustering')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#Kmeans clustering

#Eligiendo el número de clusters

distortions = []
num_clusters = range(2,10)

for i in num_clusters:
    centroids, distortion = kmeans(df[['X','Y']],i)
    distortions.append(distortion)

elbow_plot_data = pd.DataFrame({'num_clusters' : num_clusters,
                                'distortions' : distortions})

plt.figure(figsize=(25,18))
sns.lineplot(data = elbow_plot_data, x = 'num_clusters', y = 'distortions')
plt.title('Elbow method for K-Means number of clusters')

centroids,_ = kmeans(df, 4)
df['kmeans_labels'], _ = vq(df,centroids)

plt.figure(figsize=(25,18))
sns.scatterplot(data = df, x = 'X', y = 'Y', hue = 'kmeans_labels', palette = ['#133468', '#E4AC15', '#E415C5', '#860404'])
plt.title('K-Means Clustering')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)




