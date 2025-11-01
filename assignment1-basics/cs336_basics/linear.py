# linear.py
'''
权重初始化符合正态分布: 均值 mean = 0, 标准差std = sqrt(2/(d_in + d_out)
分布: N(mean, std^2), 截断区间 [-3*std, 3*std]
torch.nn.init.trunc_normal_ 实现截断正态分布

torch 行存储向量: y = xW^T, x 是行优先 R^(1 x d_in)
数学推导使用 x 列向量 y = Wx, x 是列优先 R^(d_in)
这两种表达式: W 都是 row-major，R^(d_out x d_in)，代码中也是需要行优先，注意
在

'''

import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
	def __init__(self, in_features: int, out_feature: int, device: torch.device | None = None, dtype: torch.dtype|None = None):
		'''
		in_features: 输入的维度
		out_features: 输出的维度
		device: 权重参数存储的设备
		dtype: 权重参数存储的类型
		'''
		super().__init__()
		self.in_features = in_features
		self.out_feature = out_feature
		self.weight = nn.Parameter(torch.empty((out_feature, in_features), device = device, dtype = dtype))
		std = math.sqrt(2./(in_features + out_feature))
		torch.nn.init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3 * std, b = 3 * std)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")		
