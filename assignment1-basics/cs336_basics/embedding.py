# embedding.py
'''
权重参数符合正态分布: mean = 0, std: =1, 截断区间 [-3, 3]
'''

import torch
import torch.nn as nn


class Embedding(nn.Module):
	def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device|None = None, dtype: torch.dtype|None = None):
		'''
		num_embeddings: 词表大小
		embedding_dim: 维度
		device: 权重参数设备
		dtype: 权重参数类型
		'''
		super().__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype))
		torch.nn.init.trunc_normal_(self.weight, mean = 0, std = 1, a = -3, b = 3)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 直接使用类似数组查找的方式
		return self.weight[x]