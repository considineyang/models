import torch 


x = torch.tensor([[1, 2],
                [3, 4],
                [5, 6], 
                [7, 8]])
print(x.shape)

x = x.unsqueeze(-1)

print(x)
print(x.shape)
