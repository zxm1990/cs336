#train_loop.py

import numpy as np
import os
import torch
import argparse
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import wandb

from .transformer import TransformerLM
from .adamw import MyAdamw, get_lr_cosine_schedule, gradient_clipping
from .cross_entropy import cross_entropy
from .data_loader import get_batch
from .check_point import save_checkpoint, load_checkpoint

def compute_gradient_norms(model: TransformerLM) -> dict[str, float]:
	"""
	计算所有参数的梯度范数。
	在 loss.backward() 之后、gradient_clipping 之前调用。
	"""
	grad_norms = {}
	total_norm = 0.0
	
	# 全局梯度范数
	param_norms = []
	for name, param in model.named_parameters():
		if param.grad is not None:
			param_norm = param.grad.data.norm(2)
			total_norm += param_norm.item() ** 2
			param_norms.append((name, param_norm.item()))
	
	total_norm = total_norm ** (1. / 2)
	grad_norms["grad/total_norm"] = total_norm
	
	# 各层梯度范数
	# Embedding层
	if hasattr(model.token_embeddings, 'weight') and model.token_embeddings.weight.grad is not None:
		grad_norms["grad/embedding"] = model.token_embeddings.weight.grad.norm(2).item()
	
	# 各Transformer层
	for i, layer in enumerate(model.layers):
		layer_prefix = f"grad/layer_{i}"
		
		# Attention层
		if hasattr(layer.attn, 'q_proj') and layer.attn.q_proj.weight.grad is not None:
			grad_norms[f"{layer_prefix}/attn_q"] = layer.attn.q_proj.weight.grad.norm(2).item()
		if hasattr(layer.attn, 'k_proj') and layer.attn.k_proj.weight.grad is not None:
			grad_norms[f"{layer_prefix}/attn_k"] = layer.attn.k_proj.weight.grad.norm(2).item()
		if hasattr(layer.attn, 'v_proj') and layer.attn.v_proj.weight.grad is not None:
			grad_norms[f"{layer_prefix}/attn_v"] = layer.attn.v_proj.weight.grad.norm(2).item()
		if hasattr(layer.attn, 'output_proj') and layer.attn.output_proj.weight.grad is not None:
			grad_norms[f"{layer_prefix}/attn_out"] = layer.attn.output_proj.weight.grad.norm(2).item()
		
		# FFN层
		if hasattr(layer.ffn, 'w1') and layer.ffn.w1.weight.grad is not None:
			grad_norms[f"{layer_prefix}/ffn_w1"] = layer.ffn.w1.weight.grad.norm(2).item()
		if hasattr(layer.ffn, 'w2') and layer.ffn.w2.weight.grad is not None:
			grad_norms[f"{layer_prefix}/ffn_w2"] = layer.ffn.w2.weight.grad.norm(2).item()
		if hasattr(layer.ffn, 'w3') and layer.ffn.w3.weight.grad is not None:
			grad_norms[f"{layer_prefix}/ffn_w3"] = layer.ffn.w3.weight.grad.norm(2).item()
	
	# 第一层和最后一层的梯度
	if len(model.layers) > 0:
		first_layer = model.layers[0]
		if hasattr(first_layer.attn, 'q_proj') and first_layer.attn.q_proj.weight.grad is not None:
			grad_norms["grad/first_layer_attn"] = first_layer.attn.q_proj.weight.grad.norm(2).item()
		
		last_layer = model.layers[-1]
		if hasattr(last_layer.ffn, 'w2') and last_layer.ffn.w2.weight.grad is not None:
			grad_norms["grad/last_layer_ffn"] = last_layer.ffn.w2.weight.grad.norm(2).item()
	
	# LM head
	if hasattr(model.lm_head, 'weight') and model.lm_head.weight.grad is not None:
		grad_norms["grad/lm_head"] = model.lm_head.weight.grad.norm(2).item()
	
	return grad_norms

def compute_weight_norms(model: TransformerLM) -> dict[str, float]:
	"""
	计算所有参数的权重范数。
	"""
	weight_norms = {}
	
	# Embedding层
	if hasattr(model.token_embeddings, 'weight'):
		weight_norms["weight/embedding"] = model.token_embeddings.weight.norm(2).item()
	
	# 各Transformer层
	for i, layer in enumerate(model.layers):
		layer_prefix = f"weight/layer_{i}"
		
		# Attention层
		if hasattr(layer.attn, 'q_proj') and hasattr(layer.attn.q_proj, 'weight'):
			weight_norms[f"{layer_prefix}/attn_q"] = layer.attn.q_proj.weight.norm(2).item()
		if hasattr(layer.attn, 'k_proj') and hasattr(layer.attn.k_proj, 'weight'):
			weight_norms[f"{layer_prefix}/attn_k"] = layer.attn.k_proj.weight.norm(2).item()
		if hasattr(layer.attn, 'v_proj') and hasattr(layer.attn.v_proj, 'weight'):
			weight_norms[f"{layer_prefix}/attn_v"] = layer.attn.v_proj.weight.norm(2).item()
		if hasattr(layer.attn, 'output_proj') and hasattr(layer.attn.output_proj, 'weight'):
			weight_norms[f"{layer_prefix}/attn_out"] = layer.attn.output_proj.weight.norm(2).item()
		
		# FFN层
		if hasattr(layer.ffn, 'w1') and hasattr(layer.ffn.w1, 'weight'):
			weight_norms[f"{layer_prefix}/ffn_w1"] = layer.ffn.w1.weight.norm(2).item()
		if hasattr(layer.ffn, 'w2') and hasattr(layer.ffn.w2, 'weight'):
			weight_norms[f"{layer_prefix}/ffn_w2"] = layer.ffn.w2.weight.norm(2).item()
		if hasattr(layer.ffn, 'w3') and hasattr(layer.ffn.w3, 'weight'):
			weight_norms[f"{layer_prefix}/ffn_w3"] = layer.ffn.w3.weight.norm(2).item()
	
	# LM head
	if hasattr(model.lm_head, 'weight'):
		weight_norms["weight/lm_head"] = model.lm_head.weight.norm(2).item()
	
	return weight_norms

def setup_activation_hooks(model: TransformerLM, activations: dict) -> list:
	"""
	设置 hook 来监控激活值。返回 hook handles 列表，用于后续移除。
	"""
	handles = []
	
	def make_hook(name: str):
		def hook_fn(module, input, output):
			if isinstance(output, torch.Tensor):
				activations[f"act/{name}/norm"] = output.norm(2).item()
				activations[f"act/{name}/max"] = output.max().item()
				activations[f"act/{name}/min"] = output.min().item()
				activations[f"act/{name}/mean"] = output.mean().item()
				activations[f"act/{name}/std"] = output.std().item()
		return hook_fn
	
	# 监控 embedding 输出
	handles.append(model.token_embeddings.register_forward_hook(make_hook("embedding")))
	
	# 监控各 Transformer 层的输出
	for i, layer in enumerate(model.layers):
		# Attention 输出
		handles.append(layer.attn.register_forward_hook(make_hook(f"layer_{i}/attn")))
		# FFN 输出
		handles.append(layer.ffn.register_forward_hook(make_hook(f"layer_{i}/ffn")))
		# TransformerBlock 输出
		handles.append(layer.register_forward_hook(make_hook(f"layer_{i}/output")))
	
	# 监控最终层
	handles.append(model.ln_final.register_forward_hook(make_hook("ln_final")))
	
	return handles

def load_tokenized_data(path: str | os.PathLike) -> np.ndarray:
	"""
	使用 memmap 加载 .npy 文件，避免将整个文件加载到内存。
	手动读取文件头并创建 memmap 比 np.load 的 mmap_mode 更快（约14倍）。
	虽然只在训练开始时调用一次，但考虑到训练时间较长，这点优化仍然有价值。
	"""
	# 读取文件头获取shape和dtype信息
	with open(path, 'rb') as f:
		version = np.lib.format.read_magic(f)
		if version == (1, 0):
			shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
		else:
			raise ValueError(f"Unsupported .npy file version: {version}")
		header_len = f.tell()
	
	# 创建 memmap，跳过文件头
	return np.memmap(path, mode='r', dtype=dtype, shape=shape, offset=header_len)

def get_device():
	if torch.backends.mps.is_available() and torch.backends.mps.is_built():
		return "mps"
	elif torch.cuda.is_available():
		return "cuda:0"
	else:
		return "cpu"

def train(
	# 训练集
	train_data_path: str | os.PathLike,
	valid_data_path: str | os.PathLike,

	# 模型参数
	vocab_size: int = 10000,
	d_model: int = 512,
	num_heads: int = 16,
	num_layers: int = 4,
	d_ff: int = 1344,
	theta: float = 10000.0,
	max_seq_len: int = 512,

	# 输入参数
	batch_size: int = 32,
	context_length: int = 256,

	# 优化器参数
	beta1: float = 0.9,
	beta2: float = 0.95,
	eps: float = 1e-8,
	weight_decay: float = 0.1,

	# 余弦退火参数
	min_lr: float = 6e-5,
	max_lr: float = 6e-4,
	warmup_iters: int = 1000,
	max_iterations: int = 5000,

	# 梯度裁剪
	grad_clip_norm: float = 1.0,

	# log 记录
	log_interval: int = 100,

	# 评估记录
	eval_interval: int = 1000,
	eval_batches: int = 100,

	# checkpoint 参数
	checkpoint_dir: str | os.PathLike = "checkpoints",
	checkpoint_interval: int = 1000,
	resume_from: str | os.PathLike = None,

	# 实验日志参数
	experiment_name: str = None):

	device = get_device()
	# 设置随机性
	torch.manual_seed(42)
	np.random.seed(42)

	# 创建保存点
	checkpoint_dir = Path(checkpoint_dir)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	# 初始化实验日志
	start_time = time.time()
	
	# 初始化 W&B (推荐方式 - 可以在 W&B 界面直接观察曲线)

	if experiment_name is None:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		experiment_name = f"experiment_{timestamp}"
			
	wandb.init(
		entity="hexiaomi-huster",
		project="cs336-01",
		name=experiment_name,
		config={
			"vocab_size": vocab_size,
			"d_model": d_model,
			"num_heads": num_heads,
			"num_layers": num_layers,
			"batch_size": batch_size,
			"max_lr": max_lr,
			"min_lr": min_lr,
			"max_iterations": max_iterations,
		}
	)
	# 数据加载
	train_data = load_tokenized_data(train_data_path)
	valid_data = load_tokenized_data(valid_data_path)

	# 创建model
	model = TransformerLM(vocab_size, d_model, num_heads, num_layers, d_ff,
		theta, max_seq_len, device, torch.float32)
    
	# 使用torch.compile加速训练
	if device == "cpu":
		model = torch.compile(model)
		print("✅ 使用 torch.compile (CPU)")
	elif device == "mps":
		# MPS设备：尝试使用eager backend（torch 2.6.0不支持Inductor）
		model = torch.compile(model, backend="eager")
		print("✅ 使用 torch.compile with eager backend (MPS)")
	elif device.startswith("cuda"):
		model = torch.compile(model)
		print("✅ 使用 torch.compile (CUDA)")
	num_params = sum(p.numel() for p in model.parameters())
	print(f"Model parameters: {num_params}")

	# 创建优化器
	adamw_optimizer = MyAdamw(model.parameters(), max_lr, (beta1, beta2), eps, weight_decay)

	# 如果可能，需要从checkpoint加载数据
	start_iteration = 0
	if resume_from is not None:
		start_iteration = load_checkpoint(resume_from, model, adamw_optimizer)
		print(f"Resumed from checkpoint at iteration {start_iteration}")

	# 设置训练模式
	model.train()

	# 设置激活值监控（可选，降低频率以减少开销）
	monitor_activations = False  # 设为 True 启用激活值监控
	activations = {}
	activation_handles = []
	if monitor_activations:
		activation_handles = setup_activation_hooks(model, activations)

	# 设置进度条
	pbar = tqdm(range(start_iteration, max_iterations), desc = "Training",
		initial = start_iteration, total = max_iterations, ncols = 100)

	# for 开启训练步数
	log_count = 0
	running_loss = 0.0
	running_perplexity = 0.0
	for iteration in pbar:
		# 获取数据
		inputs, targets = get_batch(train_data, batch_size, context_length, device)

		# 前向传播
		logits = model(inputs)

		# 算loss
		loss = cross_entropy(logits, targets)

		# 优化器梯度归零
		adamw_optimizer.zero_grad()

		# 反向传播
		loss.backward()

		# 计算梯度范数（在 gradient_clipping 之前）
		grad_norms = compute_gradient_norms(model)
		
		# 梯度裁剪
		gradient_clipping(model.parameters(), grad_clip_norm)

		# 更新学习率
		lr = get_lr_cosine_schedule(iteration, max_lr, min_lr, warmup_iters, max_iterations)
		adamw_optimizer.param_groups[0]['lr'] = lr

		# 优化器更新梯度
		adamw_optimizer.step()
		
		if device == "mps":
			torch.mps.synchronize()

		perplexity = torch.exp(loss).item()
		running_loss += loss.item()
		running_perplexity += perplexity

		log_count += 1
		pbar.set_postfix({
			"loss": f"{loss.item():.4f}",
			"perplexity": f"{perplexity:.2f}",
			"lr": f"{lr:.2e}"
			})

		# 记录日志
		if iteration % log_interval == 0:
			avg_loss = running_loss / log_count
			avg_perplexity = running_perplexity / log_count
			elapsed_time = time.time() - start_time
			
			# 计算权重范数（降低频率，避免开销过大）
			weight_norms = compute_weight_norms(model) if iteration % (log_interval * 10) == 0 else {}
			
			print(f"iterations:{iteration} | Loss:{avg_loss:.4f} | Perplexity:{avg_perplexity:.2f} | LR:{lr:.2e} | Time:{elapsed_time:.2f}s")
			
			# 打印梯度范数信息（如果梯度范数异常）
			if "grad/total_norm" in grad_norms:
				total_grad_norm = grad_norms["grad/total_norm"]
				if total_grad_norm > 100 or total_grad_norm < 1e-6:
					print(f"  ⚠️  梯度范数异常: {total_grad_norm:.6f}")
			
			# 记录到 W&B (在 W&B 界面观察曲线)
			log_dict = {
				"train/loss": avg_loss,
				"train/perplexity": avg_perplexity,
				"train/learning_rate": lr,
				"iteration": iteration,
				"elapsed_time": elapsed_time
			}
			
			# 添加梯度范数到日志
			log_dict.update(grad_norms)
			
			# 添加权重范数到日志（降低频率）
			if weight_norms:
				log_dict.update(weight_norms)
			
			# 添加激活值到日志（如果启用）
			if monitor_activations and activations:
				log_dict.update(activations)
				activations.clear()  # 清空，下次迭代重新填充
			
			wandb.log(log_dict, step=iteration)

			log_count = 0
			running_loss = 0.0
			running_perplexity = 0.0

		# 记录评估状态
		if iteration % eval_interval == 0 and iteration > 0:
			print(f"\nEvaluate iteration:{iteration}")
			eval_pbar = tqdm(range(eval_batches), desc = "Evaluate", leave = False, ncols = 80)
			eval_loss = evaluate(model, valid_data, batch_size, context_length, device, eval_batches, eval_pbar)
			eval_pbar.close()
			eval_perplexity = np.exp(eval_loss)
			elapsed_time = time.time() - start_time
			
			print(f"\n Valid Loss:{eval_loss:.4f} | Perplexity:{eval_perplexity:.2f} | Time:{elapsed_time:.2f}s")
			
			wandb.log({
				"valid/loss": eval_loss,
				"valid/perplexity": eval_perplexity,
				"iteration": iteration,
				"elapsed_time": elapsed_time
			}, step=iteration)

			model.train()

		# checkpoint
		if iteration % checkpoint_interval == 0 and iteration > 0:
			checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.bin"
			save_checkpoint(model, adamw_optimizer, iteration, checkpoint_path)
			print(f"Save checkpoint to {checkpoint_path}")

	pbar.close()

	checkpoint_path = checkpoint_dir / f"checkpoint_final.bin"
	save_checkpoint(model, adamw_optimizer, iteration, checkpoint_path)
	print(f"Training Complete Save checkpoint to {checkpoint_path}")
	
	# 移除激活值监控 hook
	for handle in activation_handles:
		handle.remove()
	
	wandb.finish()
	
	total_time = time.time() - start_time
	print(f"Total training time: {total_time:.2f}s")



def evaluate(model: TransformerLM, data: np.ndarray, batch_size: int, context_length: int, device: str,
	num_batches: int = 100, pbar: tqdm = None) -> float:
	# 开启评估模式
	model.eval()
	total_loss = 0.0

	# 开启无梯度计算模型
	with torch.no_grad():
		for iteration in range(num_batches):
			# 获取数据
			inputs, targets = get_batch(data, batch_size, context_length, device)
			# 前向传播
			logits = model(inputs)

			# 计算损失
			loss = cross_entropy(logits, targets)
			total_loss += loss.item()

			if pbar:
				pbar.set_postfix({"eval_loss": f"{loss.item():.4f}"})
	if device == "mps":
		torch.mps.synchronize()
	
	return total_loss / num_batches 


def main():
	parser = argparse.ArgumentParser(description="train a LLM")

	# 训练数据集
	parser.add_argument("--train-data", type = str, required = True, help = "Path to training tokenized data")
	parser.add_argument("--valid-data", type = str, required = True, help = "Path to validation tokenized data")

	# 模型参数
	# embedding
	parser.add_argument("--vocab-size", type=int, default = 10000, help = "Vocabulary size")
	parser.add_argument("--d-model", type = int, default = 512, help = "Model dimension")

	# attention
	parser.add_argument("--num-heads", type = int, default = 16, help = "Number of attention heads")
	parser.add_argument("--num-layers", type = int, default = 4, help = "Number of attention layers")

	# FFN
	parser.add_argument("--d-ff", type = int, default = 1344, help = "Feedforward dimension")

	# RoPE
	parser.add_argument("--theta", type = float, default = 10000.0, help = "RoPE theta parameter")
	parser.add_argument("--max-seq-len", type = int, default = 512, help = "Max context length")

	#训练参数
	parser.add_argument("--batch-size", type = int, default = 32, help = "Batch size")
	parser.add_argument("--context-length", type=int, default=256, help="Context length")

	# 优化器参数 beta1 beta2 eps weight-decay
	parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
	parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
	parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
	parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")

	# 余弦退火参数 max-lr min-lr warmup-itertion max-itertion
	parser.add_argument("--max-lr", type=float, default=6e-4, help="Maximum learning rate")
	parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate")
	parser.add_argument("--max-iterations", type=int, default=5000, help="Maximum number of iterations")
	parser.add_argument("--warmup-iters", type=int, default=1000, help="Warmup iterations")

	# 梯度裁剪 grad-clip-norm
	parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm")

	# 评估参数
	parser.add_argument("--eval-interval", type=int, default=1000, help="Evaluation interval")
	parser.add_argument("--eval-batches", type=int, default=100, help="Number of batches for evaluation")

	# checkpoint 参数
	parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
	parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Checkpoint interval")
	parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")

	# 实验日志参数
	parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
	parser.add_argument("--experiment-name", type=str, required = True, help="Experiment name for logging")

	args = parser.parse_args()
	
	train(
		train_data_path = args.train_data, 
		valid_data_path = args.valid_data,
		vocab_size = args.vocab_size,
		d_model = args.d_model,
		num_heads = args.num_heads,
		num_layers = args.num_layers,
		d_ff = args.d_ff,
		theta = args.theta,
		max_seq_len = args.max_seq_len,
		beta1 = args.beta1,
		beta2 = args.beta2,
		eps = args.eps,
		weight_decay = args.weight_decay,
		grad_clip_norm = args.grad_clip_norm,
		max_lr = args.max_lr,
		min_lr = args.min_lr,
		warmup_iters = args.warmup_iters,
		max_iterations = args.max_iterations,
		batch_size = args.batch_size,
		context_length = args.context_length,
		eval_interval = args.eval_interval,
		eval_batches = args.eval_batches,
		checkpoint_dir = args.checkpoint_dir,
		checkpoint_interval = args.checkpoint_interval,
		resume_from = args.resume_from,
		experiment_name = args.experiment_name,
	)

if __name__ == "__main__":
	main()