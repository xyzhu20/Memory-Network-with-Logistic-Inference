import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math, copy, time
import torch.nn.utils.rnn as rnn_utils
import numpy as np


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention' """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # (batch_size, head, Q or K length, dim/head)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class CoAttention(nn.Module):
    """DUMA"""

    def __init__(self, head, input_dim, dropout=0.3):
        super(CoAttention, self).__init__()
        self.mha = MultiHeadedAttention(head, input_dim, dropout)
        self.ff = PositionwiseFeedForward(d_model=input_dim, d_ff=input_dim * 2)
        self.norm = LayerNorm(input_dim)

    def forward(self, states1, states2, mask1, mask2):
        mask1 = mask1.unsqueeze(1)
        mask2 = mask2.unsqueeze(1)
        x1 = self.mha(states1, states2, states2, mask2)
        attn_mask1 = mask1.transpose(1, 2).expand_as(x1)  # (bsz,slen, slen)
        x1 = self.norm(self.ff(x1))

        x2 = self.mha(states2, states1, states1, mask1)
        attn_mask2 = mask2.transpose(1, 2).expand_as(x2)  # (bsz,slen, slen)
        x2 = self.norm(self.ff(x2))

        x1 = torch.mean(x1, dim=1)
        x2 = torch.mean(x2, dim=1)
        return x1, x2
