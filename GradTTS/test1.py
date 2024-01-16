import sys
import torch
from einops import rearrange

q = torch.randn((2, 4, 2, 3, 4))
k = torch.randn((2, 4, 2, 4, 4))

#attn = torch.bmm(q, k)


context = torch.einsum('bhkdn,bhken->bhkde', q, k)

qkv = torch.randn((2, 18, 4, 4))
q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)',
                    heads=2, qkv=3)
print(q.shape, k.shape, v.shape)