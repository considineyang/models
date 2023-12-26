# Transformer Encoder and Decoder Models 
# https://nn.labml.ai/transformers/models.html
import math 
import torch
import torch.nn as nn

from labml_nn.utils import clone_module_list
from labml_nn.transformers.feed_forward import FeedForward
from MHA import MultiHeadAttention
from labml_nn.transformers.positional_encoding import get_positional_encoding

# Embed tokens and add fixed positional encoding
class EmbeddingsWithPositionalEncoding(nn.Module):

    def __init_(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[:x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe
    
# Embed tokens and add parameterized positional encodings
class EmbeddingsWithLearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, n_vocab: int, max_len: int=5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[:x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe
    
# Transformer Layer
class TransformerLayer(nn.Module):

    def __init__(self, *, 
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention = None,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        # src_attn is the self sttention module when this is used in a decoder
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_self_attn = nn.LayerNorm([d_model])
        # Whether to save input to the feed forward layer
        self.is_save_ff_input = False

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(self_attn)
        # If a source is provided, get results from attention to source, this is 
        # when you have a decoder layer that pays attention to encoder outputs
        if src is not None:
            z = self.norm_src_attn(x)
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(attn_src)
        z = self.norm_ff(x)
        if self.is_save_ff_input:
            self.ff_input = z.clone()
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x
    
# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return self.norm(x)

# Transformer Decoder
class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        return self.norm(x)
    
# Generator 
# This predicts the tokens and gives the lof softmax of those. 
# You don't need this if you are using nn.CrossEntropyLoss .        
class Generator(nn.Module):

    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return self.projection(x)

# Combined Encoder-Decoder
class EncoderDecoder(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: nn.Module, 
                 tgt_embed: nn.Module, 
                 generator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

# This was important from their code. Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        enc = self.encode(src, src_mask)
        return self.decode(enc, src_mask, tgt, tgt_mask)
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)






