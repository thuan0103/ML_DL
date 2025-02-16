import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('segmented_customers.csv')

# print(df.describe())

# plt.figure(1, figsize=(15,6))
# n = 0
# for x in ['Age','Annual Income (k$)','Spending Score (1-100)']:
#     n += 1
#     plt.subplot(1,3,n)
#     plt.subplots_adjust(hspace = 0.5 , wspace=0.5)
#     sns.histplot(df[x], bins = 15)
#     plt.title('Distplot of {}'.format(x))
# plt.show()

data = df[['Annual Income (k$)','Spending Score (1-100)']].values
print(data.shape[0])

def eucidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

def kmeans(data, max_iteration = 100, k =3):
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0],k, replace=False)]
    for _ in range(max_iteration):
        clusters = [[] for _ in range(k)]
        for point in data:
            print(point)
            distance = [eucidean_distance(point,centroid) for centroid in centroids]
            clusters_idx = np.argmin(distance)
            clusters[clusters_idx].append(point)

        new_centroids = np.array([ np.mean(cluster, axis=0) if cluster else cluster[i] for i, cluster in enumerate(clusters)])

        if np.all(centroids==new_centroids):
            break

        centroids = new_centroids
    return centroids, clusters

centroids, clusters = kmeans(data)

cluster_labels = np.zeros(data.shape[0])
for i,cluster in enumerate(clusters):
    for point in cluster:
        idx = np.where((data == point).all(axis=1))[0][0]
        cluster_labels[idx] = i

df['Cluster'] = cluster_labels.astype(int)

print(df)