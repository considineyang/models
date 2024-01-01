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

T1 = torch.Tensor([[[1,2,3],[4,5,6],[7,8,9]],
                  [[1,2,3],[4,5,6],[7,8,9]],
                  [[1,2,3],[4,5,6],[7,8,9]]])
T = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
print(T.shape)
def shift_right(x: torch.Tensor):
    print(x)
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    print(zero_pad)
    x_padded = torch.cat([x, zero_pad], dim=1)
    print(x_padded)
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    print(x_padded)
    x = x_padded[:-1].view_as(x)
    print(x)
    return x

shift_right(T)