# transformer.py

import torch
import torch.nn as nn
import math
from .rmsnorm import RMSNorm
from .attention import MultiSelfAttention
from .swiglu import FFN
from .linear import Linear
from jaxtyping import Float, Int
from .embedding import Embedding

def init_lnear_weight(out_feature: int, in_features: int, 
	device: torch.device|None = None, dtype: torch.dtype|None = None) -> torch.Tensor:
	weight = nn.Parameter(torch.empty((out_feature, in_features), device = device, dtype = dtype))
	std = math.sqrt(2./(in_features + out_feature))
	torch.nn.init.trunc_normal_(weight, mean = 0.0, std = std, a = -3 * std, b = 3 * std)
	return weight


class TransformerBlock(nn.Module):
	def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int,
		device: torch.device|None = None, dtype: torch.dtype|None = None):
		super().__init__()
		self.ln1 = RMSNorm(d_model, device = device, dtype = dtype)
		self.attn = MultiSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
		self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)
		self.ffn = FFN(d_model, d_ff, device, dtype)

	def forward(self, x: Float[torch.Tensor, "batch sequence_length d_model"],
		token_positions: Int[torch.Tensor, "... sequence_length"] | None = None) -> Float[torch.Tensor, "batch sequence_length d_model"]:
		if token_positions is None:
			token_positions = torch.arange(x.size(1), device = x.device).expand(x.size(0), -1)
		x = x + self.attn(self.ln1(x), token_positions)
		x = x + self.ffn(self.ln2(x))
		return x

class TransformerLM(nn.Module):
	def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
	 	d_ff: int, theta: float, max_seq_len: int,
		device: torch.device|None = None, dtype: torch.dtype|None = None):
		super().__init__()

		self.num_layers = num_layers
		self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
		# 必须使用 nn.ModuleList, 否则会丢失参数管理
		self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len, device, dtype) for _ in range(num_layers)])
		self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
		self.lm_head = Linear(d_model, vocab_size, device, dtype)

	def forward(self, in_features: Float[torch.Tensor, "batch sequence_length"]) -> Float[torch.Tensor, "... batch sequence_length vocab_size"]:
		x = self.token_embeddings(in_features)
		token_positions = torch.arange(in_features.size(1), device = in_features.device).expand(in_features.size(0), -1)
		for layer in self.layers:
			x = layer(x, token_positions)

		x = self.ln_final(x)
		out = self.lm_head(x)
		return out




