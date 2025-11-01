# 架构实践
其他调试模型架构的良好实践包括：
开发神经网络架构时，通常第一步是让模型在单个小批量数据上过拟合。若实现正确，应能快速将训练损失降至接近零。
在模型各组件设置调试断点，检查中间张量的形状是否符合预期。
监控激活值、模型权重和梯度的范数，确保未出现梯度爆炸或消失问题
1. 测试模型是否过拟合
使用小批量数据 batch_size = 2, context_length=32，较大的学习率 max_lr = 1e-3
准备一个固定的batch数据，训练 50 - 200步，观察 loss 迅速接近 0
 uv run python -m cs336_basics.train_loop --train-data=data/TinyStoriesV2-GPT4-train.npy --valid-data=data/TinyStoriesV2-GPT4-valid.npy --experiment-name="cs336-01-00" --batch-size=2 --context-length=32 --max-iterations=200 --max-lr=1e-3
    1. 发现有时候loss出现了负数，这不符合预期，观察交叉熵损失函数，发现targets里面有tokenid 大于10000
    2. 回溯数据，发现取数据的时候，确实取到了21980的tokenid，但是在 BPE 进行encode时，并没有产生 21980的token id，最后发现 mmap 读数据没有越过头，导致读取数据出错
    3. 为什么target都超过了数组，交叉熵损失函数没有出现奔溃呢？mps 设备下，torch数组越界不报错

F9 在关键行设置断点
F5 启动调试
F10 单步执行，不进入函数
F11 进入感兴趣的函数
Shift+F11 快速跳出
在调试控制台输入表达式查看值
F5 继续执行到下一个断点

模块执行：一般是目录下有 __init__ 文件，uv run python -m ....
F5 默认是文件调试，如果要以模块调试，使用添加配置

观察:
token_embedding 输出shape 应该是 [batch, seq_len, d_model]
TransformerBlock.forward 输入输出 shape 一致
lm_head 输出 shape：应该是 [batch, seq_len, vocab_size]

3. 监控激活值、权重和梯度的范数
    目的：尽早发现梯度爆炸/消失、激活值异常、权重异常。

激活值范数监控
    需要监控的地方（你项目中）：
        每个 TransformerBlock 的输出（各层激活值）
        Attention 的输出
        FFN 的输出
        lm_head 的输入（最终特征）

监控指标：
    L2 范数（torch.norm(x)）：衡量激活值的整体大小
    最大值/最小值：检查是否有异常大的激活值
    均值/标准差：检查激活值分布
预期行为：
    各层激活值范数在合理范围（例如 0.1–100）
    避免出现极大值（> 1000）或极小值（< 1e-6）

权重范数监控
需要监控的地方：
token_embeddings.weight
每个 TransformerBlock 中的：
attn 的 Q/K/V/O 权重
ffn 的 w1/w2/w3 权重
lm_head.weight
监控指标：
每个参数的 L2 范数
初始化后的初始范数
训练过程中范数的变化趋势
预期行为：
初始化后范数在合理范围（由你的 init_lnear_weight 决定）
训练过程中缓慢变化，不会突然跳跃
梯度范数监控
需要监控的地方（你项目中）：
反向传播之后，在 gradient_clipping 之前
各层参数的梯度
特别是：
Embedding 层的梯度
第一层和最后一层的梯度（通常较大/较小）
监控指标：
全局梯度范数：所有参数梯度的总体范数
每层梯度范数：比较不同层的梯度大小
梯度裁剪前的范数（你已经做了 gradient_clipping）
预期行为：
梯度范数在合理范围（例如 0.01–10）
如果梯度范数 > 100，可能是梯度爆炸
如果梯度范数 < 1e-6，可能是梯度消失
在你的项目中的实现方式
在 train_loop.py 中的合适位置：
在 loss.backward() 之后，gradient_clipping() 之前检查梯度范数
在关键层（如第一层和最后一层）记录激活值范数
使用 W&B 记录这些指标，便于观察曲线
