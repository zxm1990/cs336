'''
1. 先使用 bpe 训练，训练的 merges 和 vocab 写入文件
2. 通过 merges 和 vocal 创建一个tokenizer
3. 使用新的 tokenizer 来encode和decoder


训练BPE的结论：
训练 TinyStoriesV2-GPT4-valid，耗时:  BPE 耗时: 0.9127750396728516 秒, 最长 token: '================' (长度: 16)
训练集(TinyStoriesV2-GPT4-train.txt) BPE tokenizer 时间: 70.48761296272278 秒, 最长 token: 'sssssssssssssssssssssssss
sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
ssssssssssssssssssssssss' (长度: 512)
owt_valid 训练验证集 BPE 耗时: 77.01747798919678 秒, 最长 token: '------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
----------------------------------' (长度: 1024)


'''

import json
import time
import pathlib
from .adapters import run_train_bpe

DATA_DIR = (pathlib.Path(__file__).parent.parent) / "data"
OUTPUT_DIR = (pathlib.Path(__file__).parent.parent) / "output"


def get_longest_token_info(vocab):
    """获取词汇表中最长 token 的信息"""
    longest_token = max(vocab.values(), key=lambda x: len(x.decode('utf-8', errors='replace')))
    longest_token_str = longest_token.decode('utf-8', errors='replace')
    return longest_token_str, len(longest_token_str)


def save_vocab_and_merges(vocab, merges, output_dir):
    """将词汇表和合并规则保存到文件"""
    output_dir.mkdir(exist_ok=True)
    
    # 将 bytes 转换为字符串，并按照参考格式保存：token -> id
    vocab_str = {token_bytes.decode('utf-8', errors='replace'): token_id 
                 for token_id, token_bytes in vocab.items()}
    
    with open(output_dir / "vocab.json", "w") as f:
        json.dump(vocab_str, f, indent=4, ensure_ascii=False)
    
    with open(output_dir / "merges.txt", "w") as f:
        for merge in merges:
            # 将 bytes 转换为字符串
            token1 = merge[0].decode('utf-8', errors='replace')
            token2 = merge[1].decode('utf-8', errors='replace')
            f.write(f"{token1} {token2}\n")


def test_train_bpe_tinystories_valid():
    """训练验证集的 BPE tokenizer"""
    input_path = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
    start_time = time.time()
    
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    
    end_time = time.time()
    longest_token_str, longest_length = get_longest_token_info(vocab)
    print(f"训练验证集 BPE 耗时: {end_time - start_time} 秒, 最长 token: '{longest_token_str}' (长度: {longest_length})")
    output_dir = OUTPUT_DIR / "tinystories_valid"
    save_vocab_and_merges(vocab, merges, output_dir)


def test_train_bpe_tinystories_train():
    """训练训练集的 BPE tokenizer"""
    input_path = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    
    end_time = time.time()
    longest_token_str, longest_length = get_longest_token_info(vocab)
    print(f"训练集(TinyStoriesV2-GPT4-train.txt) BPE tokenizer 时间: {end_time - start_time} 秒, 最长 token: '{longest_token_str}' (长度: {longest_length})")
    output_dir = OUTPUT_DIR / "tinystories_train"
    save_vocab_and_merges(vocab, merges, output_dir)

def test_train_bpe_owt_valid():
    """训练验证集的 BPE tokenizer"""
    input_path = DATA_DIR / "owt_valid.txt"
    start_time = time.time()
    
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    
    end_time = time.time()
    longest_token_str, longest_length = get_longest_token_info(vocab)
    print(f"训练验证集 BPE 耗时: {end_time - start_time} 秒, 最长 token: '{longest_token_str}' (长度: {longest_length})")
    output_dir = OUTPUT_DIR / "owt_valid"
    save_vocab_and_merges(vocab, merges, output_dir)

# def test_train_bpe_owt_train():
#     """训练验证集的 BPE tokenizer"""
#     input_path = DATA_DIR / "owt_train.txt"
#     start_time = time.time()
    
#     vocab, merges = run_train_bpe(
#         input_path=input_path,
#         vocab_size=32000,
#         special_tokens=["<|endoftext|>"],
#     )
    
#     end_time = time.time()
#     longest_token_str, longest_length = get_longest_token_info(vocab)
#     print(f"训练训练集 BPE 耗时: {end_time - start_time} 秒, 最长 token: '{longest_token_str}' (长度: {longest_length})")
#     output_dir = OUTPUT_DIR / "owt_train"
#     save_vocab_and_merges(vocab, merges, output_dir)

