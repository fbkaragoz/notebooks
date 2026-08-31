import math

import torch
import torch.nn as nn


class CSA(nn.modules):  # causal self-attention
    def __init__(self, dim):
        super().__init__()

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)  # query
        k = self.k_proj(x)  # key
        v = self.v_proj(x)  # value

        scores = q @ k.transpose(-2, 1)  # every q compares itself with every k

        scores = scores / math.sqrt(C)  # scale

        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        out = weights @ v  # blend values

        return out
