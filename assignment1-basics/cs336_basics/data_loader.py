# data_loader.py
'''

'''

import numpy.typing as npt
import torch
import numpy as np


def get_batch(data_set: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        生成 batch_size 批数据，每个seq长度为 context_length, 起始位置随机，虽然不同的seq的内容有重复
        这是LLM运行的，因为 LLM 是多轮重复迭代
        '''
        max_start_idx = len(data_set) - context_length
        # 生成 batch_size 个数据，
        start_indices = np.random.randint(0, max_start_idx, size = batch_size)

        # input_sequences = []
        # target_sequences = []
        # for start_idx in start_indices:
        #     input_seq = data_set[start_idx: start_idx + context_length]
        #     input_sequences.append(input_seq)

        #     target_seq = data_set[start_idx+1: start_idx + 1 + context_length]
        #     target_sequences.append(target_seq)

        # # 转换为 numpy 数组，以前是 list，每个list里面也是array，现在转换为2维array
        # input_sequences = np.array(input_sequences, dtype = np.int64)
        # target_sequences = np.array(target_sequences, dtype = np.int64)

        # # 转换为 tensor
        # input_tensor = torch.from_numpy(input_sequences).to(device)
        # target_tensor = torch.from_numpy(target_sequences).to(device)
        input_tensor = torch.stack(
            [
                # 将 numpy 类型先转换为 np.int64 再转换为 tensor，主要是避免 embedding 词表的问题
                # 毕竟还不知embedding层需要适配所有的词表大小
                torch.from_numpy((data_set[start_idx: start_idx + context_length]).astype(np.int64))
                for start_idx in start_indices
            ]
        )
        target_tensor = torch.stack(
            [
                torch.from_numpy(data_set[start_idx+1: start_idx + 1 + context_length].astype(np.int64))
                for start_idx in start_indices
            ]
        )
        if "cuda" in device:
            input_tensor = input_tensor.pin_memory().to(device, non_blocking=True)
            target_tensor = target_tensor.pin_memory().to(device, non_blocking=True)
        else:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

        return input_tensor, target_tensor