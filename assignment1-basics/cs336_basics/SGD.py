# SGD.py

import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

class MySGD(torch.optim.Optimizer):
	def __init__(self, params, lr = 1e-3):
		defaults = {"lr": lr}
		super().__init__(params, defaults)

	def step(self, closure: Optional[Callable] = None):
		'''
		closure: 对损失进行重新计算的闭包函数
		return:
			标量，返回损失值
		'''
		loss = None if closure is None else closure()

		for group in self.param_groups:
			lr = group["lr"]

			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p]
				t = state.get("t", 0)
				grad = p.grad.data
				# 学校率随着训练步数衰减
				p.data -= lr/math.sqrt(t+1) * grad
				state["t"] = t + 1


		return loss


# 创建可训练参数 (10x10随机矩阵，初始标准差为5)
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

# 初始化SGD优化器(学习率设为1)
opt = MySGD([weights], lr=1e3)

# 训练循环
for t in range(10):
    opt.zero_grad()                  # 重置所有可训练参数的梯度
    loss = (weights**2).mean()       # 计算标量损失值(示例使用L2正则)
    print(loss.cpu().item())         # 打印当前损失值
    loss.backward()                  # 反向传播计算梯度
    opt.step()                       # 执行优化器更新

'''
1. lr = 1e1
29.997549057006836
19.19843101501465
14.152263641357422
11.072635650634766
8.968833923339844
7.436191558837891
6.271440505981445
5.359124660491943
4.628026008605957
4.031525135040283
发现loss 是越来越小的。

2. lr = 1e2
24.288053512573242
24.288049697875977
4.16717004776001
0.0997297465801239
1.2212507798622701e-16
1.3611597157586983e-18
4.5835037400320796e-20
2.730424651222377e-21
2.3423345198317347e-22
2.6025941212870577e-23
loss 在最小值附近震荡

3. lr = 1e3
26.233177185058594
9470.17578125
1635648.625
181948416.0
14737821696.0
930125316096.0
47749588320256.0
2054391842471936.0
7.572050535671398e+16
2.4314693595126825e+18
loss 直接错过最小值

'''
