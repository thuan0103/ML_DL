import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('segmented_customers.csv')

data = torch.tensor(df[['Annual Income (k$)','Spending Score (1-100)']].values,dtype=torch.float32).to(device=device)

def k_means(data,k=3, max_interation = 100):
    torch.manual_seed(42)
    centroids = data[torch.randperm(data.shape[0])[:k]]

    for _ in range(max_interation):
        distances = torch.cdist(data,centroids)
        
        cluster_idx = torch.argmin(distances, dim=1)

        new_centroids = torch.stack([data[cluster_idx==i].mean(dim=0) if (cluster_idx == i).sum() > 0 else centroids[i] for i in range(k)])

        if torch.all(centroids == new_centroids):
            break

        centroids = new_centroids
    
    return centroids, cluster_idx

centroids, cluster_labels = k_means(data)
df['cluster'] = cluster_labels.cpu().numpy().astype(int)

print(df)