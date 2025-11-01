# generate_text.py
'''
温度缩放，v_i/t, 当 t < 1 时，会放大概率值，这样导致确定性的概率越大，生成文本
更稳定，当然可能是稳定的好，也可能是稳定的差
当 t > 1 时，每个概率都分布差不多，这样选择的更多，生成的稳定不确定性就越大

top P 采样: 截断低概率的样本，重新算概率分布

'''
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path
from .transformer import TransformerLM
from .tokenizer import BPETokenizer
from .train_loop import get_device
from .check_point import load_checkpoint

DATA_DIR = (Path(__file__).parent.parent) / "data"


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0):
	if temperature < 0:
		raise ValueError("Temperature must positive")

	return logits / temperature

def top_p_sampling(logits: torch.Tensor, p: float = 0.9):
	# 算概率
	probs = F.softmax(logits, dim=-1)
	# 概率排序
	sorted_probs, sorted_indices = torch.sort(probs, descending=True)
	# 算累和概率
	cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
	# 剔除低概率
	mask = cumsum_probs <= p
	mask[0] = True

	# 过滤并重新归一化，简单除法，不会改变分配比例
	filtered_probs = sorted_probs * mask.float()
	filtered_probs = filtered_probs / filtered_probs.sum()

	# 随机采样
	sampled_index = torch.multinomial(filtered_probs, num_samples = 1)

	tokenid = sorted_indices[sampled_index].item()
	return tokenid

def generate(model: TransformerLM, tokenizer: BPETokenizer, prompt: str, max_tokens: int = 500,
	temperature: float = 1.0, top_p: float = 0.9, endoftext_token: str = "<|endoftext|>"):
	device = get_device()
	inputs = tokenizer.encode(prompt)
	input_tensor = torch.tensor([inputs], dtype=torch.int32, device=device)
	endof_tokenid = None
	if endoftext_token in tokenizer.special_tokens:
		endof_tokenid = tokenizer.encode(endoftext_token)[0]

	model.eval()
	generated_tokens = []
	with torch.no_grad():
		for _ in range(max_tokens):
			logits = model(input_tensor)
			next_token_logits = logits[0, -1, :]
			next_token_logits = apply_temperature(next_token_logits, temperature=temperature)
			next_tokenid = top_p_sampling(next_token_logits, p=top_p)

			if next_tokenid == endof_tokenid:
				break

			generated_tokens.append(next_tokenid)
			next_tensor = torch.tensor([[next_tokenid]], dtype=torch.int32, device=device)
			input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

	all_token_ids = inputs + generated_tokens
	generated_text = tokenizer.decode(all_token_ids)
	return generated_text

def load_tokenizer(vocab_path: str, merges_path: str):
	with open(vocab_path, 'rb') as f:
		vocab = pickle.load(f)
	
	with open(merges_path, 'rb') as f:
		merges = pickle.load(f)
	
	return BPETokenizer.new_instance(vocab, merges, ['<|endoftext|>'])

if __name__ == "__main__":
	device = get_device()
	tokenizer = load_tokenizer(f"{DATA_DIR}/TinyStoriesV2-GPT4-vocab.pkl", f"{DATA_DIR}/TinyStoriesV2-GPT4-merges.pkl")
	model = TransformerLM(vocab_size=10000, d_model=512, num_heads=16,
		num_layers=4, d_ff=1344, theta=10000.0, max_seq_len=512, device=device, dtype=torch.float32)

	checkpoint_path = Path("checkpoints/checkpoint_final.bin")
	if checkpoint_path.exists():
		load_checkpoint(checkpoint_path, model, None)
		print(f"Loaded checkpoint from {checkpoint_path}")

	model.to(device)
	prompt = "Once upon a time there was a little boy named Ben"
	output = generate(model, tokenizer, prompt)
	print(output)


