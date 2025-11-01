Q: Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?

A:
参数分析：
embedding: vocab_size x d_mode = 80411200
TransformerBlock:
ln1: d_model = 1600
attention: d_model x d_model x 4 = 10240000
ln2: d_model = 1600
ffn: d_ffn x d_model x 2 + d_model x d_ffn = 30720000
总和: 48 x (1600 +  10240000 + 1600 + 30720000) = 1966233600

lnFinal: 1600
llm_head: d_model x vocab_size = 80411200

最终参数: 2,127,057,600 = 2.127B
加载内存: 2.127 x 4 = 8.5G 

FLOPS分析：
embedding: 直接取的数组，没有乘法运算
RMSNorm: x_i / norm(x_i) * scale ，逐元素相乘，不涉及乘法运算
attention:
Q: (seq_len x d_model) x (d_model x d_model) = 2sd^2
K: 2sd
V: 2sd
= 2 * 1024 x 1600 x 1600 = 15,728,640,000
Q^T x K = (seq_len x d_model) x (d_model x seq_len) = 2ds^2
多头: (n, s, h_d) x (h_d, s, n) = n x 2 x h_d x s^2 = 2ds^2 = 2 x 1600 x 1024 x 1024
 = 3,355,443,200
S x V = (n, s, s) x (s, h_d, n) = n x 2 x h_d x s x s= 2ds^2 = 3,355,443,200
a x O = (s, d) x (d d) = 2sd^ = 5,242,880,000
总和: 5,242,880,000 x 4 + 3,355,443,200 x 2 = 27,682,406,400

ffn:
w1: (s, d) x (d, d_ff) = 2dsd_ff
w3: (s, d) x (d, d_ff) = 2dsd_ff
w2: (s, d_ff) x (d_ff, d) = 2d_ffsd
总和: 6 x 1600 x 6400 x 1024 = 62,914,560,000

transformerBlock:
48 x (27,682,406,400 + 62,914,560,000) = 4,348,654,387,200

llm_head:
(s, d) x (d vocab_size) = 2 x 1024 x 1600 x 50,257 = 164,682,137,600
 总和大约： 4.5 TFLOPs

 消耗最多的是 ffn 占比在 70% 左右

GPT-2 Small (12层，768 d_model，12头):
TransformerBlock: 
3 x 2sd^2 + 2ds^2 x 2 + 2sd^2 = 8sd^2 + 4ds^2
ffn: 6dsd_ff 

12 x (8sd^2 + 4ds^2 + 6dsd_ff)

llm_head: 

当 seq_len = 16,384 时，计算量：149.5 TFLOPs
长度增加16倍，计算量增加了33倍，此时 FFN计算占比从 67% 下降到 32%，
attention(不包含QKV投影)计算从 7.1% 增加到 55%