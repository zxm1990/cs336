# adamw.py
'''
初始化参数: θ
初始化一阶矩: m = 0
初始化二阶矩: n = 0
梯度可以简单理解为斜率，在数学中已经有了方向和大小，那么为什么这里还需要 m 和 v 呢？
梯度计算并不是所有训练数据的梯度，是取一部分 mini-batch 数据的梯度，每个批次算的梯度
不一样，而且是随机采样的数据来算的，所以得到的梯度有噪声。
只使用单一梯度，那么 m 由于符号不同，大小不同，会削弱梯度的大小，保留是整体的趋势
v 的计算是梯度平方，相当于忽略符号，只保留大小。所以最后算学习率的系数需要：m/sqrt(v)

为什么需要调整学习率？
一般 b_1 = 0.9 , b_2 = 0.999
首次 m = 0.1g_1 ; v = 0.001g_1^2 ，在更新参数时，m 本应该更靠近 g，结果偏离g
因此需要对 m 进行纠偏 m = m/(1 - b_1) 由于每一步都要纠偏 m = m/(1 - b_1^t)
同理 v 也是如此。
α = α * m/sqrt(v)
  = α * ( (m/(1-b_1^t)) / sqrt(v/(1-b_2^t)))
  = α * sqrt(1-b_2^t)/(1-b_1^t)) * m/sqrt(v);

于是得到了调整学习率 α_t 的公式

权重衰减:
1. 不论梯度是否变化，由于有了权重衰减，参数一定会变小
2. 防止了参数变大，参数的数字绝对值较大，会出现过拟合（在训练数据上拟合非常好，但是在测试数据上，一点微小变化出现结果较大偏差）
3. 注意权重衰减不在受梯度的影响

fot t = 1 ... T:
    g = p.grad
    m <-- b_1 * m + (1 - b_1) * g
    v <-- b_2 * v + (1 - b_2) * g^2
    
    # 调整学习率
    α_t <-- α * (sqrt(1 -(b_2)^t) / (1 - (b_1)^t)
    
    # 参数更新
    θ <-- θ - α_t * (m / (sqrt(v) + ϵ)

    # 权重衰减，
    θ <-- θ - αλθ

1. 问题一: 
 运行AdamW需要多少峰值内存？根据参数、激活值、梯度和优化器状态的内存使用情况回答。答案应以batch_size和模型超参数
 （vocab_size、context_length、num_layers、d_model、num_heads）表示。假设d_ff = 4 × d_model。
为简化计算，在考虑激活值内存使用时仅计算以下组件：
Transformer模块
RMSNorm(s)
多头自注意力子层：QKV投影、QᵀK矩阵乘法、softmax、值的加权求和、输出投影
位置前馈网络：W₁矩阵乘法、SiLU激活、W₂矩阵乘法
最终RMSNorm
输出嵌入层
logits的交叉熵计算

交付要求：分别给出参数、激活值、梯度和优化器状态的代数表达式，以及总内存表达式
参数内存：
1. embedding: [vocab_size, d_mode] ==> vocab_size x d_model = VD
2. RMSNorm: a_i / RMS(a) * g_i==> D
3. Q: [d_model, d_model / num_heads] ==> DD/H, 同理 K 和 V 一样
O: [d_model, d_model] ===> DD
attention: 3DD/H * H + DD = 4DD
4. 进入 FFN 之前，需要一个 RMSNorm: D
5: FFN: W1[d_model, d_ff], w3[d_modle, d_ff], w2[d_ff, d_modle] ===> 3 x 4D x D = 12 D^2
6: BlockTransformer: 4D^2 + 2D + 12D^2 = L(16D^2 + 2D)
7: RMSNorm: D
8: llm_head: [d_model, vocab_size] ===> VD
总内存: 2VD + D + L(16 D^2 + 2D)

激活值内存: 这里不讨论激活值是否需要保存，只统计计算过程中激活值的内存
1. embedding: input[B, S, V] x [V, D] = [B, S, D]
2. transformerBlock: 
    2.1 RMSNorm: [B, S, D] ===> [B, S, D]
    2.2 Q: [B, S, D] x [, D] ===> [B, H, S, D/H], 同理K，V 也是这样 ===> 3BSD
    2.3 Score: Q^T x K /sqrt(d_k) ==> [B, H, S, D/H] x [B, H, S, D/H]^T = [B, H, S, S] = BHS^2
    2.4 weight: softmax(score) ==> [B, H, S, S] ==> BHS^2
    2.5 attention: [B, H, S, S] x [B, H, S, D/H] ==> [B, H, S, D/H] ===> [B, S, D]
    2.6 O: [B, S, D] x [D, D] = BSD

    总结: 6BSD + 2BHS^2

    2.6: RMSNorm: [B, S, D] ==> [B, S, D]
    2.7: W1X * W3X ==> [B, S, D_ff]
    2.8: sigLu: [B, S, D_ff]
    2.9: W2X: [B, S, D]
    总结: 2BSD + 2BSD*4 = 10 BSD
(2BHS^2 + 16BSD) x L
3. RMSNorm: BSD
4. llm_head: BSV

激活值内存: 2BSD + BSV + L(16BSD + 2BHS^2)

梯度内存: 一个参数一个梯度, 与参数表达式一样，2VD + D + L(16 D^2 + 2D)
优化器状态: m 与 参数shape一样，v 和参数 shape 一样

总内存表达式: 4(2VD + D + L(16 D^2 + 2D)) + 2BSD + BSV + L(16BSD + 2BHS^2)
= 8VD + 4D + 64LD^2 + 8LD + (2+16L)BSD + BSV + 2LBHS^2

2.问题二：
将您的答案具体化为 GPT-2 XL 模型规格，
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
得出仅与 batch_size 相关的表达式。在 80GB 内存限制下，最大可用 batch_size 是多少？
Total_elements(B)=8,508,230,400+3,829,614,668⋅B
假设全部使用 FP32, 
内存占用 = 4(8,508,230,400+3,829,614,668⋅B) = 34,032,921,600+15,318,458,672⋅B
        = 34,032,921,600+15,318,458,672⋅B
        = 34G + 15G * B
 如果内存是 80G, Batch 最大约为 3 

3: 问题三：
AdamW 单步计算耗费多少 FLOPs？
m = b_1 * m + (1 - b_1) g
2次乘法，1次加法: 3FLOPs
v = b_2 * v + (1 - b_2)g^2
3次乘法，1次加法，4FLOPs

修正
m = m/(1 - b_1 ^ t)
v = v/(1 - b_2 ^ t)
常数： 2次FLOPs

权重更新:
p = p - α_t * m/(sqrt(v) + eps)
开方1次，+eps 1次，除法1次，乘法1次，+p 1次 = 5FLOPs

权重衰减:
p = p - αλθ
1次乘法，1次加法， 2FLOPs
总共: 3 + 4 + 2 + 5 + 2 = 16Params FLOPs

问题二：
MFU: 观察到 FLOP/s / 理论 FLOP/s 
A100的理论吞吐: 19.5T FLOP/s, 假设 MFU = 50%
使用 GPT-2 XL 模型，训练 4万step，Batch_size = 1024, 在单个A100模型上需要多少天？
假设反向传播FLOPs 是前向传播的 2倍

forward：之前算过大约是 4.5T FLOPs x 1024(batch)
backward: 大约 9T FLOPs x (1024)
AdamW: 16 x 2.127B = 34.032 x 10^9
单step： 13.5 x 1024 x 10^12 FLOPs = 13.5 x 10^15 FLOPs
耗时： 13.5 x 10^15 x 4 x 10^4 / (19.5 x 10^12 /2 )
    = 13.5 x 4 x 10^7/9.85 / 86400 = 634 天

'''

import torch
from collections.abc import Callable, Iterable
from typing import Iterator, List, Dict, Tuple, Optional
from jaxtyping import Float, Int
import math

class MyAdamw(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01):
        defaults = {
            "lr":lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> torch.Tensor:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"][0], group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["step"] = 0

                state["step"] += 1
                step = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                grad = p.grad

                m.mul_(beta1).add_(grad, alpha = 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                adjusted_lr = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)

                p.data.add_(m/(v.sqrt() + eps), alpha = -adjusted_lr)

                p.data.add_(p.data, alpha = -lr * weight_decay)

            return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    '''
    lr 并不是固定不变的，使用 余弦退火调度策略
    Args:
        it: 当前迭代步数 t
        max_learning_rate: 最大学习率 α_max
        min_learning_rate: 最小学习率 α_min
        warmup_iters: 预热步数 T_w
        cosine_cycle_iters: 退火总步数 T_c
    Returns:
        lr: float 学习率

    当 t < T_w :
        α_t = t/T_w *  α_max
    当 T_w <= t <= T_c:
        α_t = α_min + 1/2 * (1 + cos((t-T_w)/(T_c - T_w) * π)) * (α_max - α_min)
    当 t > T_c:
        α_t = α_min
    '''
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    if it > cosine_cycle_iters:
        return min_learning_rate

    cos_args = math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
    return min_learning_rate + 1/2 * (1 + cos_args) * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    '''
    训练过程中，有时会遇到产生较大梯度的样本，可能会导致训练不稳定，此时可以采用梯度裁剪
    但是 **所有参数** 的梯度的 L_2 范数: ||p.grad||_2 < max_l2_norm: 不做任何修改
    当 ||p.grad||_2 >= max_l2_norm , grad 按比例缩小，max_l2_norm / (||p.grad||_2 + eps)

    Args:
        parameters: 模型参数的可迭代对象
        max_l2_norm: 最大L2范数阈值
        eps: 防止除零的小常数
    '''
    grads = []
    for p in parameters:
        if p.grad is None:
            continue
        grads.append(p.grad)    

    if len(grads) == 0:
        return

    # 计算所有梯度的总L2范数，想将所有参数的梯度组合成一个向量，再求 norm
    total_norm = torch.norm(
        torch.cat([grad.view(-1) for grad in grads]), 2
    )

    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(scale)











