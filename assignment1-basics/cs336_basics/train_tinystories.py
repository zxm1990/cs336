# train_tinystories.py

import os
import pathlib
from tokenizer import BPETokenizer
import numpy as np
import json
import pickle

DATA_DIR = (pathlib.Path(__file__).parent.parent) / "data"

def gpt2_bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup tables
    between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_tokenizer_from_file(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    tokenizer = BPETokenizer.new_instance(vocab, merges, special_tokens)
    return tokenizer

def train_bpe(input_path: str, vocab_size: int, output_path:list[str]):
    tokenizer = BPETokenizer(vocab_size, 8, ['<|endoftext|>'])
    tokenizer.train(input_path)
    if output_path:
        with open(output_path[0], "wb") as f:
            pickle.dump(tokenizer.vocab, f)
        with open(output_path[1], "wb") as f:
            pickle.dump(tokenizer.merges, f)

def encode_big_file(vocab_path: str, merges_path: str, input_path: str, output_path: str):
    with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    
    with open(merges_path, 'rb') as f:
            merges = pickle.load(f)
    
    tokenizer = BPETokenizer.new_instance(vocab, merges, ['<|endoftext|>'])
    all_ids = tokenizer.encode_file(input_path)
    np.save(output_path, np.array(all_ids, dtype=np.uint16))

if __name__ == "__main__":
    if not os.path.exists(f"{DATA_DIR}/TinyStoriesV2-GPT4-vocab.pkl"):
        train_bpe(f"{DATA_DIR}/TinyStoriesV2-GPT4-train.txt", 10000,
                  [f"{DATA_DIR}/TinyStoriesV2-GPT4-vocab.pkl", f"{DATA_DIR}/TinyStoriesV2-GPT4-merges.pkl"])
        print("完成 tinystories 的BPE训练")

    if not os.path.exists(f"{DATA_DIR}/TinyStoriesV2-GPT4-train.npy"):
        encode_big_file(f"{DATA_DIR}/TinyStoriesV2-GPT4-vocab.pkl", f"{DATA_DIR}/TinyStoriesV2-GPT4-merges.pkl",
                    f"{DATA_DIR}/TinyStoriesV2-GPT4-train.txt", f"{DATA_DIR}/TinyStoriesV2-GPT4-train.npy")
        print("完成 tinystories tokenizer")

    if not os.path.exists(f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.npy"):
        encode_big_file(f"{DATA_DIR}/TinyStoriesV2-GPT4-vocab.pkl", f"{DATA_DIR}/TinyStoriesV2-GPT4-merges.pkl",
                    f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.txt", f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.npy")
        print("完成 tinystories valid tokenizer")

