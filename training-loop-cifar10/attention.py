import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V

torch.manual_seed(0)
Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)
out = scaled_dot_product_attention(Q, K, V)
ref = F.scaled_dot_product_attention(Q, K, V)
print("shape:", out.shape)
print("match:", torch.allclose(out, ref, atol=1e-6))

mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
x = torch.randn(2, 4, 8)
out2, weights2 = mha(x, x, x)
print("mha out:", out2.shape, "weights:", weights2.shape)