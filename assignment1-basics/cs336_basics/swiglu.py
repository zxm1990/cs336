#swiglu.py
'''
采用 SwitGLU 激活函数，此函数是结合了 SiLU 和 GLU 的机制
SiLU(x) = x . a(x) = x / (1 + e^(-x))
sigmoid(x) = 1 / (1 + e^(-x) 这里避免算 SiLU 时 出现精度问题，可以直接使用 torch.sigmoid
GLU(x, W1, W2) = a(W1x) ⊙ (W3x)

结合上面2个函数，构造新的 FFN 网络

FFN(x) = SwitchLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x) 
其中 : x 是 R^(d_model), W1 和 W3 是 R^(d_ff x d_model) , W2 是 R^(d_model x d_ff)

'''
import torch
import torch.nn as nn
import math
from .linear import Linear
from einops import einsum

def SiLU(x: torch.Tensor) -> torch.Tensor:
	in_dtype = x.dtype
	x = x.to(torch.float32)
	return (x * torch.sigmoid(x)).to(in_dtype)

class SwiGLU(nn.Module):
	def __init__(self, d_model: int, d_ff: int, device: torch.device|None = None, dtype: torch.dtype|None = None):
		super().__init__()
		self.d_model = d_model
		self.d_ff = d_ff
		self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device = device, dtype = dtype))
		self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device = device, dtype = dtype))
		self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device = device, dtype = dtype))
		std = math.sqrt(2./(d_ff + d_model))
		torch.nn.init.trunc_normal_(self.w1, mean = 0., std = std, a = -3 * std, b = 3 * std)
		torch.nn.init.trunc_normal_(self.w2, mean = 0., std = std, a = -3 * std, b = 3 * std)
		torch.nn.init.trunc_normal_(self.w3, mean = 0., std = std, a = -3 * std, b = 3 * std)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
		out3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
		y = SiLU(out1) * out3
		return einsum(y, self.w2, "... d_ff, d_model d_ff -> ... d_model")


class FFN(nn.Module):
	def __init__(self, d_model: int, d_ff: int, device: torch.device|None = None, dtype: torch.dtype|None = None):
		'''
		d_model: 隐藏层维度
		d_ff: FFN 层维度
		'''
		super().__init__()
		self.w1 = Linear(d_model, d_ff, device = device, dtype = dtype)
		self.w2 = Linear(d_ff, d_model, device = device, dtype = dtype)
		self.w3 = Linear(d_model, d_ff, device = device, dtype = dtype)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.w2(SiLU(self.w1(x)) * self.w3(x))
