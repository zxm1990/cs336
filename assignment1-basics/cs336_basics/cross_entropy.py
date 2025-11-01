# cross_entropy.py
'''
DL散度:
散度数学定义: D = plog (p/q) = plogp - plogq
p 是真实分布，是固定值，所以损失函数为 L = -plogq

H(p_b_t, q_b_t) = -Sum_{i~(0, v-1)} (p_b_t_i x log(q_b_t_i))
p_b_t: 真实的概率分布；在第 b 个批次，第 t 个位置(seq中的位置)，在词表中的概率分布
q_b_t: 模型预测的概率分布
Sum_{i~(0, v-1)}: 在词表中每个token位置的求和
在语言模型中，真实分布是: i = 目标位置时，p_b_t_i = 1, 其他时候为0
则意味着
H(p_b_t, q_b_t) = -log(q_b_t_i)
此时 q_b_t_i 应该为某个真实tokenid 的softmax之后概率值

于是损失 L = -log(softmax(logits)[target]) 通过softmax给每个token在词表每个位置都算了一个概率值
普通的实现: 对每个token 先算在 vocab 维度的softmax，然后只取某个index下的概率值。
优化实现：词表的每个index都算概率，完全没有必要，因为没有被使用，只算target index下的概率值
softmax(o_i)[j] = exp(o_i[j]) / sum(exp(o_i[all]))
L = -o_i[j] + log(sum(exp(o_i[all])))

数值稳定性版本：
m = max(o_i)
L = -(o_i[j] - m) + log(sum(exp(o_i[all] - m)))

perplexity = exp(loss_mean)
有了loss，为什么还需要 perplexity呢？loss表示是概率分布差异，平均负对数似然
而 perplexity ：从多少个均匀选项中挑选一个的难度。
例如 loss = 0， perplexity = 1; 模型总能选正确，模型一点都不困惑
如果 perplexity = 100， 模型需要从100个选项中，选一个，这太难了


'''

import torch
from jaxtyping import Float, Int
from einops import reduce

def cross_entropy(
	logits: Float[torch.Tensor, "... vocab_size"],
	targets: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, ""]:
	'''
	logits: 未归一化的logits，shape一般是 [batch, seq_len, vocab_size]
	targets: 目标类别索引, shape 一般是 [batch, seq_len], 值一般是在 词表中的index

	return:
		标量损失值，一般是对 batch, seq_len 的平均
	'''

	# 使用float32 计算
	in_dtype = logits.dtype
	logits = logits.to(torch.float32)

	# 算max
	max_logits = reduce(logits, "... vocab_size -> ... 1", "max")

	# 算 log(sum(exp(o_i[all] - m)))
	logits_exp = torch.exp(logits - max_logits)
	log_sum_exp = torch.log(reduce(logits_exp, "... vocab_size -> ... 1", "sum"))

	# 找到 targets 对应的数字
	num_positions = logits.shape[:-1].numel() # 除去最后一个维度，其他所有维度的数目
	logits_flat = logits.reshape(num_positions, -1) # 展开成2维 (num_positions, vocab_size)。-1 表示字段推断维度
	targets_flat = targets.reshape(-1) # （num_positions），展开为1维，-1自动推算维度

	position_indices = torch.arange(num_positions, device = logits.device)
	# 在 mps 设备上，这里越界不会抛出异常，而是出现随机值
	targets_logits = logits_flat[position_indices, targets_flat]

	# 变成一维
	max_logits_flat = max_logits.reshape(-1)
	log_sum_exp_flat = log_sum_exp.reshape(-1)

	loss = -(targets_logits - max_logits_flat) + log_sum_exp_flat

	loss_mean = loss.mean()

	return loss_mean.to(in_dtype)

