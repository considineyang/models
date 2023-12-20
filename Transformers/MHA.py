# This is the implement of multi-headed attention 
# From paper: Attention is All You Need  
# Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html
import math 
from typing import Optional, List

import torch 
from torch import nn
from labml import tracker


""" 
This module does a linear transformation and splits the vector into given numbers
of heads for multi-head atttention, this is used to transform 
key,query and values vectors
"""

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()

        self.linear = nn.Linear(d_model, heads * d_k, bias=bias) # Linear layer for linear transform 
        self.heads = heads # Number of heads
        self.d_k = d_k # Number of dimensions in vectors in each head

    def forward(self, x: torch.Tensor):
        # input shape: [seq_len, batch_size, d_model] or [batch_size, d_model] 
        # we apply the transformation to the last dimension and split that into the heads
        head_shape = x.shape[:-1]

        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, heads: int, d_model: int, dropout_prob: float=0.1, bias: bool = True):
            super().__init__()
            self.d_k = d_model // heads
            self.heads = heads
            
            self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
            self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
            self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

            self.softmax = nn.Softmax(dim=1)
            self.output = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout_prob)
            self.scale = 1 / math.sqrt(self.d_k)
            self.attn = None # Store attentions so can be used for logging

        def get_scores(self, query: torch.Tensor, key: torch.Tensor):
            # Calculate scores between queries and keys
            return torch.einsum('ibhd,jbhd->ijbh', query, key)
        
        def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
            # mask has shape [seq_len_q, seq_len_k, batch_size]
            assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
            assert mask.shape[1] == key_shape[0]
            assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

            mask = mask.unsqueeze(-1)
            return mask # resulting mask has shape [seq_len_q, seq_len_k, batch_size, heads]
        
        def forward(self, *,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    mask: Optional[torch.Tensor] = None):
            # q,k,v have shape [seq_len, batch_size, d_model] mask has shape [seq_len, seq_len, batch_size]
            # mask[i,j,b] indicates whether for batch b, query at position i has access to key-value at j.

            seq_len, batch_size, _ = query.shape
            if mask is not None:
                mask = self.prepare_mask(mask, query.shape, key.shape)

            query = self.query(query)
            key = self.key(key)
            value = self.value(value)

            scores = self.get_scores(query, key)
            scores *= self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = self.softmax(scores)

            tracker.debug('attn', attn)
            attn = self.dropout(attn)

            x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
            self.attn = attn.detach()
            x = x.reshape(seq_len, batch_size, -1)
            return self.output(x)
        


        





