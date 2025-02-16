from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('segmented_customers.csv')

data = df[['Annual Income (k$)','Spending Score (1-100)']].values

kmeans_model = KMeans(init='random',n_clusters=3,random_state=42, n_init=100)

df['cluster'] = kmeans_model.fit_predict(data)

print(df)