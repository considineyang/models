import torch 
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention

def shift_right(x: torch.Tensor):
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])