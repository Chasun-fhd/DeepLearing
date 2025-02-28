import torch
a = torch.arange(10, dtype=torch.long)
print(a)
print(a.unsqueeze(0))