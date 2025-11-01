# check_point.py
'''
通常要保持 模型状态，优化器状态，如果使用余弦退火调度，还需保持迭代步数
通过 state_dict() 获取状态，一般将状态数据转换为 dict ，调用 torch.save(obj, out) 来保存
'''

import torch
import typing
import os

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
	iteration: int, out: str| os.PathLike | typing.BinaryIO | typing.IO[bytes]):
	obj = {
		"model_state": model.state_dict(),
		"optim_state": optimizer.state_dict(),
		"iteration": iteration
	}
	torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
		model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
		obj = torch.load(src)
		model.load_state_dict(obj["model_state"])
		if optimizer is not None:
			optimizer.load_state_dict(obj["optim_state"])
		return int(obj["iteration"])