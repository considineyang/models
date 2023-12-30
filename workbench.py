import torch 
import math

As = torch.randn(3,2,5)
Bs = torch.randn(3,5,4)
Cs = torch.einsum('bij,bjk->bik', As, Bs)

# 等价操作
torch.bmm(As, Bs)
A = torch.Tensor([[[1,2,3],
                   [2,3,4]],
                  [[1,2,3],
                   [2,3,4]]])
B = torch.Tensor([[[1,2],
                   [3,4],
                   [4,5]],
                  [[1,2],
                   [5,3],
                   [2,3]]])
C = torch.einsum('bij,bjk->bik', A, B)
print(A)
print(B)
print(C)