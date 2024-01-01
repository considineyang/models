# https://nn.labml.ai/transformers/xl/index.html
from typing import List, Optional

import torch
import torch.nn as nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list
from RelativeMHA import RelativeMultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward

class TransformerXLLayer(Module):

    def __init__(self, *,
                 d_model: int,
                 self_attn: RelativeMultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *,
                x: torch.Tensor,
                mem: Optional[torch.Tensor],
                mask: torch.Tensor):
        # x is a tensor of the token level feature vectors of shape [seq_len, batch_size, d_model]
        # mem is a tensor of the past token level feature vectors of shape [mem_len, batch_size, d_model]
        # mask is a matrix of shape [seq_len, mem_len + seq_len, batch_size] or
        # [seq_len, mem_len + seq_len, 1] . mask[i, j] is true if token at i can see token at j
        z = self.norm_self_attn(x) # normalize the vectors before doing self attention
        if mem is not None:
            mem = self.norm_self_attn(mem)
            m_z = torch.cat((mem, z), dim=0)
        else:
            m_z = z
        self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
        x = x + self.dropout(self_attn)
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x
    
class TransformerXL(Module):

    def __init__(self, layer: TransformerXLLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])
    
    def forward(self, x: torch.Tensor, mem: List[torch.Tensor], mask: torch.Tensor):
        new_mem = []
        for i, layer in enumerate(self.layers):
            new_mem.append(x.detach())
            m = mem[i] if mem else None
            x = layer(x=x, mem=m, mask=mask)
        return self.norm(x), new_mem
    