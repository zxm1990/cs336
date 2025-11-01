#softmax.py
'''
softmax = exp(x) / sum(exp^x) 避免溢出，x = x- max
'''

import torch
from einops import reduce

def softmax(x: torch.Tensor, dim:int = -1) -> torch.Tensor:
	if dim < 0:
		dim += x.ndim
	indexs = list(range(x.ndim))
	# 计算的维度交换到最后维度
	indexs[dim], indexs[-1] = indexs[-1], indexs[dim]
	x_trans = x.permute(*indexs)

	x_max = reduce(x_trans, "... n -> ... 1", "max")

	x_exp = (x_trans - x_max).exp()
	x_sum = reduce(x_exp, "... n -> ... 1", "sum")

	out = x_exp / x_sum

	return out.permute(*indexs)