import torch
import torch.nn as nn
from torchtyping import TensorType
import math

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        scores = q @ torch.transpose(k, 1, 2)
        scores = scores / math.sqrt(self.attention_dim)
        mask = torch.tril(torch.ones(scores.shape[-2], scores.shape[-1], device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = nn.functional.softmax(scores, dim=-1)
        scores = torch.round(scores @ v, decimals=4)

        return scores
