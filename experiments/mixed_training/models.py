"""Five readable univariate direct multi-step forecasting models."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DLinear(nn.Module):
    def __init__(self, context, horizon, kernel):
        super().__init__()
        if kernel % 2 != 1 or kernel > context:
            raise ValueError('DLinear kernel must be odd and <= context')
        self.kernel = kernel
        self.trend = nn.Linear(context, horizon)
        self.residual = nn.Linear(context, horizon)

    def forward(self, x):
        padded = F.pad(x.unsqueeze(1), (self.kernel//2, self.kernel//2), mode='replicate')
        trend = F.avg_pool1d(padded, self.kernel, stride=1).squeeze(1)
        return self.trend(trend)+self.residual(x-trend)


class GRU(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.gru = nn.GRU(1, 32, batch_first=True)
        self.output = nn.Linear(32, horizon)

    def forward(self, x):
        _, state = self.gru(x.unsqueeze(-1))
        return self.output(state[-1])


class Transformer(nn.Module):
    def __init__(self, context, horizon):
        super().__init__()
        width = 32
        self.input = nn.Linear(1, width)
        position = torch.arange(context).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2).float()*(-math.log(10000)/width))
        encoding = torch.zeros(context, width)
        encoding[:, 0::2] = torch.sin(position*frequency)
        encoding[:, 1::2] = torch.cos(position*frequency)
        self.register_buffer('position', encoding)
        block = nn.TransformerEncoderLayer(width, 4, dim_feedforward=64, dropout=.1,
                                           activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.output = nn.Linear(width, horizon)

    def forward(self, x):
        encoded = self.encoder(self.input(x.unsqueeze(-1))+self.position)
        return self.output(encoded[:, -1])


def build_model(name, spec):
    l, h = spec['context'], spec['horizon']
    if name == 'linear':
        return nn.Linear(l, h)
    if name == 'dlinear':
        return DLinear(l, h, spec['kernel'])
    if name == 'mlp':
        return nn.Sequential(nn.Linear(l, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, h))
    if name == 'gru':
        return GRU(h)
    if name == 'transformer':
        return Transformer(l, h)
    raise ValueError(f'Unknown model: {name}')
