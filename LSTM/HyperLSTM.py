# This a implement of hypernetwork for LSTM.
# Using a small netwoek (called a "hypernetwork") to generate the weights for a large network (main network)
# A hypernetwork is just a small network that generates the weights of a much larger network 
# embedding a small Hypernetwork inside a large LSTM
# This is a dynamic hypernetwork for LSTM, the parameters of each step are changed by a smaller LSTM network
# https://nn.labml.ai/hypernetworks/hyper_lstm.html
# https://blog.otoro.net/2016/09/28/hyper-networks/

from typing import Optional, Tuple

import torch 
from torch import nn

from labml_helpers.module import Module 
from LSTM import LSTMCell

class HyperLSTMCell(Module):

    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int):
        # n_z is the size of the feature vectors used to alter the LSTM weights
        super().__init__()
        self.hyper = LSTMCell(hidden_size + input_size, hyper_size, layer_norm=True)
        self.z_h = nn.Linear(hyper_size, 4 * n_z)
        self.z_x = nn.Linear(hyper_size, 4 * n_z)
        self.z_b = nn.Linear(hyper_size, 4 * n_z, bias=False)
        
        d_h = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_h = nn.ModuleList(d_h)

        d_x = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_x = nn.ModuleList(d_x)

        d_b = [nn.Linear(n_z, hidden_size) for _ in range(4)]
        self.d_b = nn.ModuleList(d_b)

        self.w_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(4)])
        self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(4)])

        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.layer_norm_c = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor,
                h_hat: torch.Tensor, c_hat: torch.Tensor):
        x_hat = torch.cat((h, x), dim=-1)

        h_hat, c_hat = self.hyper(x_hat, h_hat, c_hat)

        z_h = self.z_h(h_hat).chunk(4, dim=-1)
        z_x = self.z_x(h_hat).chunk(4, dim=-1)
        z_b = self.z_b(h_hat).chunk(4, dim=-1)

        ifgo = []
        for i in range(4):
            d_h = self.d_h[i](z_h[i])
            d_x = self.d_x[i](z_x[i])

            y = d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + \
            d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + \
            self.d_b[i](z_b[i])
            ifgo.append(self.layer_norm[i](y))

        i, f, g, o = ifgo
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next, h_hat, c_hat

class HyperLSTM(Module):
    
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, n_layers: int):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size

        self.cells = nn.ModuleList([HyperLSTMCell(input_size, hidden_size, hyper_size, n_z)] +
                                  [HyperLSTMCell(hidden_size, hidden_size, hyper_size, n_z)] for _ in range(n_layers - 1))
    
    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]] = None):
        n_steps, batch_size = x.shape[:2]

        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            h_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
            c_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]


        else: 
            (h, c, h_hat, c_hat) = state
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
            h_hat, c_hat = list(torch.unbind(h_hat)), list(torch.unbind(c_hat))

        out = []
        for t in range(n_steps):
            inp = x[t]
            for layer in range(self.n_layers):
                h[layer], c[layer], h_hat[layer], c_hat[layer] = self.cells[layer](inp, h[layer], c[layer], h_hat[layer], c_hat[layer])
                inp = h[layer]

            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)
        h_hat = torch.stack(h_hat)
        c_hat = torch.stack(c_hat)

        return out, (h, c, h_hat, c_hat)


