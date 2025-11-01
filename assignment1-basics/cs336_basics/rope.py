# rope.py
'''
每个位置 i 的token，都有 q^i = W_q @ x^i, 形状是 R^d 这里的 i 是指在 序列中的位置
使用一个成对的旋转矩阵: q^i' = R^i @ W_q @ x^i, R 是对 d x d 的对角矩阵（对角线有值，其他位置都是0），
R^i_k = cos(theta_i,k),  -sin(theta_i,k)
		sin(theta_imk),   cos(theta_i,k)
旋转的角度 theta_i,k = i / 10000^{(2k-1)/d} 其中 i 是位置，d是维度，10000是常数， k 的取值是 [1,2,3 ... d/2]
这里 2k-1 全是奇数，实现过程中可以换成 k/(d/2), 简单一点
k越小，则 theta 越大，频率大，则变化慢，也就是低纬

来分析一下 e 这个常数，e 越大，则 theta 越小，那么？？？
可以看到 R 只与 i 和 k 有关，k是固定的取值，范围在 [1,2,3 ... d/2]。i 虽然是但是，在不同的batch中，i 取值范围
也是有限的，那么在跨layer 和 跨batch，是不是需要重复使用 R
'''
import torch
import torch.nn as nn
from einops import einsum, rearrange


class RoPE(nn.Module):
	def __init__(self, theta: float, d_k: int, max_seq_len:int, device: torch.device|None = None):
		super().__init__()
		self.theta = theta
		self.d_k = d_k
		self.max_seq_len = max_seq_len

		half_dim = d_k // 2
		k_range = torch.arange(half_dim, dtype = torch.float32, device = device)
		theta_range = 1./(self.theta ** (k_range / half_dim))

		pos_range = torch.arange(max_seq_len, dtype = torch.float32, device = device)

		# 频率矩阵，外积构造超大矩阵, 每个位置都有一行向量，表达每个维度的旋转角度
		freqs = einsum(pos_range, theta_range, "i, j -> i j")
		cos = torch.cos(freqs)
		sin = torch.sin(freqs)

		# 存储到缓存中
		self.register_buffer("cos_cache", cos)
		self.register_buffer("sin_cache", sin)

	def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
		in_dtype = x.dtype
		x = x.to(torch.float32)

		x_pair = rearrange(x, "... seq_len (d_pair two) -> ... seq_len d_pair two", two = 2)
		cos = self.cos_cache[token_position]
		sin = self.sin_cache[token_position]

		cos = rearrange(cos, "... s d -> ... 1 s d")
		sin = rearrange(sin, "... s d -> ... 1 s d")

		x1, x2 = x_pair.unbind(dim = -1)
		rotary1 = x1 * cos - x2 * sin
		rotary2 = x1 * sin + x2 * cos

		rotary = torch.stack((rotary1, rotary2), dim = -1)

		out = rearrange(rotary, "... seq_len d_pair two -> ... seq_len (d_pair two)", two = 2)

		return out.to(in_dtype)


