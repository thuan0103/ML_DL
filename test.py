import torch

a = torch.tensor([[60, 59]], dtype=torch.float32)
b = torch.tensor([[20,12]], dtype=torch.float32)

c = torch.cdist(a,b)
print(c)