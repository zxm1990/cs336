#attention.py

import torch
import torch.nn as nn
from einops import einsum, rearrange
from .softmax import softmax
import math
from jaxtyping import Float, Int
from .rope import RoPE
from .linear import Linear

def scaled_dot_product_attention(
	Q: Float[torch.Tensor, "... queries d_k"],
	K: Float[torch.Tensor, "... keys d_k"],
	V: Float[torch.Tensor, "... values d_v"],
	mask: Float[torch.Tensor, "... queries keys"] | None = None):
	
	d_k = Q.shape[-1]
	scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(~mask, float("-inf"))
	weights = softmax(scores, dim = -1)
	out = einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
	return out

class MultiSelfAttention(nn.Module):
	def __init__(self, d_model: int, num_heads: int,
		theta: float | None = None, 
		max_seq_len: int | None = None,
		device: torch.device|None = None,
		dtype: torch.dtype|None = None):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.theta = theta
		self.max_seq_len = max_seq_len
		
		head_dim = self.d_model // num_heads
		self.rope = None
		if theta is not None and max_seq_len is not None:
			self.rope = RoPE(theta, head_dim, max_seq_len, device)

		self.q_proj = Linear(d_model, d_model, device, dtype)
		self.k_proj = Linear(d_model, d_model, device, dtype)
		self.v_proj = Linear(d_model, d_model, device, dtype)
		self.output_proj = Linear(d_model, d_model, device, dtype)


	def forward(self, in_features: Float[torch.Tensor, "... seq_len d_in"],
		token_positions: Int[torch.Tensor, " ... seq_len"] | None = None,) -> Float[torch.Tensor, "... seq_len d_model"]:

		seq_len = in_features.shape[-2]
		Q = rearrange(self.q_proj(in_features), "... seq_len (num_head k_head_dim) -> ... num_head seq_len k_head_dim", num_head = self.num_heads)
		K = rearrange(self.k_proj(in_features), "... seq_len (num_head k_head_dim) -> ... num_head seq_len k_head_dim", num_head = self.num_heads)
		V = rearrange(self.v_proj(in_features), "... seq_len (num_head v_head_dim) -> ... num_head seq_len v_head_dim", num_head = self.num_heads)

		if self.rope:
			Q = self.rope(Q, token_positions)
			K = self.rope(K, token_positions)

		# 如果使用 triu, 那么后续直接使用mask，而且此时需要主对角线往上一行 diagonal = 1
		mask = torch.tril(torch.ones(seq_len, seq_len, device = Q.device, dtype = torch.bool), diagonal = 0)
		attention = scaled_dot_product_attention(Q, K, V, mask)
		attention = rearrange(attention, "... head seq_len d_v -> ... seq_len (head d_v)")

		return self.output_proj(attention)
