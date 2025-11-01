# rmsnorm.py
'''
RMSNorm: 均方根层归一化
对于一个维度 d_model 的向量 a，RMSNorm会对每个 a_i 都进行一个缩放训练:
RMSNorm(a_i) = a_i / RMS(a) * g_i； 每个值先除以一个均方根，再乘以一个参数，
其中 g_i 是可训练的，也是维度 d_model 的向量
RMS(a) = sqrt(sum(a_i ^ 2) / d_model + e): 每个元素平方和再求平均，加上一个超参数，之后再开方
e: 超参数，默认是 1e-5

避免精度溢出: 计算时，需要先将其转换为 float32
in_dtype = x.dtype
x = x.to(torch.float32)
...

return result.to(in_dtype)

可以看到，RMSNorm 是对维度中每个数字进行一个缩放，缩放之前先Norm下

'''

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
	def __init__(self, d_model: int,  eps: float = 1e-5, device: torch.device|None = None, dtype: torch.dtype|None = None):
		super().__init__()
		'''
		d_model: 隐藏层维度
		eps: 超参数
		'''
		self.d_model = d_model
		self.eps = eps
		# 初始化为1，首次不缩放
		self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		in_dtype = x.dtype
		x = x.to(torch.float32)
		# 使用 torch.mean 求矩阵在某个维度的均值
		norm = torch.sqrt(torch.mean(x**2, dim = -1, keepdim = True) + self.eps)
		result = x / norm * self.weight
		
		return result.to(in_dtype)