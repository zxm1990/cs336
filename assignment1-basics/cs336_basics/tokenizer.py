# tokenizer.py 
'''
1. Unicode 就是给每个独一无二的字符(中文，日文，英文等)编码出唯一的码点(数字)，目前大约有 15万+
2. utf-8: 对唯一的码点进行字节编码，可以使用1字节，2字节，最多使用4字节来表示
3. 为什么倾向于在 utf-8 编码的字节上训练分词器？
utf-8: 英文表示1字节，压缩效果好，没有端序问题
utf-16: 大多数表示都是2字节，英文也是2字节
utf-32: 定长编码，固定4字节表示
4. 提供一个示例证明下面函数错误
def decode_utf8_bytes_to_str_wrong(bytesting: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
这个函数任务每个字节都能解码为Unicode字符，看起来英文执行的很好，但是中文必定出错，因为大多数中文都需要3个字节
5. 给出一个不能解码为任何Unicode字符的两字节序列
使用中文进行 utf-8 encode 之后，直接取前面2个字节，必定不能解码为任何 Unicode 字符

性能分析，使用 cProfile 分析
1. 生成性能文件
uv run python -m cProfile -o profile_output.prof -m pytest tests/test_train_bpe.py::test_train_bpe_speed -v
参数说明：
python -m cProfile： 启动性能分析器
-m: 模块方式运行，这里同时运行了 cProfile 和 pytest
-m pytest: 启动pytest
-v: 详细输出

2. 查看性能文件
uv add snakeviz
uv run snakeviz profile_output.prof

'''


import regex as re
import os
from typing import Iterable, Iterator, List, Dict, Tuple
from multiprocessing import Pool
from collections import Counter, defaultdict
import heapq
try:
    from .pretokenization_example import find_chunk_boundaries
except ImportError:
    from pretokenization_example import find_chunk_boundaries


def _encode_chunk(args):
    chunk, vocab, merges, special_tokens = args
    tokenizer = BPETokenizer.new_instance(vocab, merges, special_tokens)
    return tokenizer.encode(chunk)


GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)


def pretokenize(text: str) -> list[bytes]:
	str_blocks = re.findall(GPT2_SPLIT_PATTERN, text)
	str_bytes = [ s.encode("utf-8") for s in str_blocks]
	return str_bytes

def iter_pretokenize(text: str) -> Iterator[bytes]:
	for it in GPT2_RE.finditer(text):
		yield it.group(0).encode("utf-8")


def word_count(args:tuple[str, List[str], re.Pattern]) -> Counter:
	chunk, special_tokens, special_pattern = args[0], args[1], args[2]
	word_list = []
	if special_pattern:
		text_parts = special_pattern.split(chunk)
	else:
		text_parts = [chunk]

	for text in text_parts:
		if not text or text in special_tokens:
			continue
		for word in pretokenize(text):
			if word:
				word_ids = tuple(word) # word 已经是 bytes 对象
				word_list.append(word_ids)
	# 统计每个bytes对象的次数
	return Counter(word_list)

class Node:
	def __init__(self, value: tuple, count: int):
		self.value = value
		self.count = count
		self.pre = None
		self.next = None

class SortItem:
	def __init__(self, count: int, id_pair: Tuple[int, int], bytes_pair: Tuple[bytes, bytes]):
		self.count = count
		self.bytes_pair = bytes_pair
		self.id_pair = id_pair

	def __lt__(self, other: 'SortItem'):
		if self.count != other.count:
			return self.count > other.count
		return self.bytes_pair > other.bytes_pair

class BPETokenizer:
	def __init__(self, vocab_size: int, num_chunk: int, special_tokens: list[str] | None = None):
		self.__vocab_size = vocab_size
		self.__num_chunk = num_chunk
		self.__special_tokens = special_tokens or []
		self.__special_token_bytes = [token.encode("utf-8") for token in self.__special_tokens]

		self.__bytes_to_ids: Dict[bytes, int] = {}
		self.__id_to_bytes: Dict[int, bytes] = {}
		self.__merges: List[tuple[bytes, bytes]] = []

		# special_token 分配 0 开始的id
		for i, token in enumerate(self.__special_token_bytes):
			self.__bytes_to_ids[token] = i
			self.__id_to_bytes[i] = token

		# utf-8 一定是单字节编码，所以先把单字节id占用了
		offset = len(self.__special_token_bytes)
		for i in range(256):
			self.__bytes_to_ids[bytes([i])] = i + offset
			self.__id_to_bytes[i + offset] = bytes([i])

		# 词表: 根据 id 查词
		self.__vocab = self.__id_to_bytes.copy()

	def train(self, path: str | os.PathLike):
		'''
		性能优化版本:
		1. 所有的pair的排序，使用大顶堆，直接获取最高次数的pair
		2. 只更新由于合并引发的pair的次数，使用双向链表来组合pair
		'''

		assert self.__vocab_size >= len(self.__vocab)
		all_word_counter = self.__collect_all_word(path)

		pair_postions, pair_counter, pq = self.__group_all_tokens(all_word_counter)

		merge_nums = self.__vocab_size - len(self.__vocab)
		for i in range(merge_nums):
			best_pair = self.__get_best_pair(pq, pair_counter)
			if not best_pair:
				break
			p1, p2 = best_pair
			new_token_id = len(self.__bytes_to_ids)
			new_bytes = self.__id_to_bytes[p1] + self.__id_to_bytes[p2]
			self.__bytes_to_ids[new_bytes] = new_token_id
			self.__id_to_bytes[new_token_id] = new_bytes
			self.__merges.append((self.__id_to_bytes[p1], self.__id_to_bytes[p2]))

			# 更新变化的双向链表和优先级队列
			pos_nodes = pair_postions[best_pair]
			pos_nodes_list = list(pos_nodes) # 避免迭代过程中集合变化，导致迭代失败
			for node in pos_nodes_list:
				node1 = node
				node2 = node.next
				if not node2:
					continue

				count = node1.count

				# 更新前面pair的计数
				if node1.pre:
					left_node = node1.pre
					old_left_pair = (left_node.value, node1.value)
					pair_counter[old_left_pair] -= count
					heapq.heappush(pq, SortItem(pair_counter[old_left_pair], old_left_pair,
						(self.__id_to_bytes[old_left_pair[0]], self.__id_to_bytes[old_left_pair[1]])))
					pair_postions[old_left_pair].discard(left_node)

					# 产生新的合并
					new_left_pair = (left_node.value, new_token_id)
					pair_counter[new_left_pair] += count
					pair_postions[new_left_pair].add(left_node)
					heapq.heappush(pq, SortItem(pair_counter[new_left_pair], new_left_pair,
						(self.__id_to_bytes[new_left_pair[0]], self.__id_to_bytes[new_left_pair[1]])))

				#更新后面pair的计数
				if node2.next:
					right = node2.next
					old_right_pair = (node2.value, right.value)
					pair_counter[old_right_pair] -= count
					pair_postions[old_right_pair].discard(node2)
					heapq.heappush(pq, SortItem(pair_counter[old_right_pair], old_right_pair,
						(self.__id_to_bytes[old_right_pair[0]], self.__id_to_bytes[old_right_pair[1]])))

					# 产生新的合并
					new_right_pair = (new_token_id, right.value)
					pair_counter[new_right_pair] += count
					pair_postions[new_right_pair].add(node1)
					heapq.heappush(pq, SortItem(pair_counter[new_right_pair], new_right_pair, 
						(self.__id_to_bytes[new_right_pair[0]], self.__id_to_bytes[new_right_pair[1]])))

				# 更新本节点
				node1.value = new_token_id
				node1.next = node2.next # node2 已经消失了
				if node2.next:
					node2.next.pre = node1

			del pair_counter[best_pair]
			del pair_postions[best_pair]

		# 更新词表
		self.__vocab = self.__id_to_bytes.copy()


	def encode(self, text: str) -> list[int]:
		if not text:
			return []
		special_pattern = self.__special_pattern()
		if not special_pattern:
			return self.__encode(text)

		all_token_ids = []
		text_parts = special_pattern.split(text)
		for t in text_parts:
			if t in self.__special_tokens:
				all_token_ids.append(self.__bytes_to_ids[t.encode("utf-8")])
			else:
				all_token_ids.extend(self.__encode(t))

		return all_token_ids

	def encode_file(self, path: str | os.PathLike) -> list[int]:
		chunks = []
		with open(path, "rb") as f:
			boundaries = find_chunk_boundaries(f, self.__num_chunk, b"<|endoftext|>")
			for start, end in zip(boundaries[:-1], boundaries[1:]):
				f.seek(start)
				chunk = f.read(end - start).decode("utf-8", errors="ignore")
				chunks.append(chunk)
		
		vocab = self.vocab
		merges = self.merges
		special_tokens = self.__special_tokens
		
		all_token_ids = []
		with Pool(processes=self.__num_chunk) as pool:
			results = pool.imap_unordered(_encode_chunk, [(chunk, vocab, merges, special_tokens) for chunk in chunks])
			for res in results:
				all_token_ids.extend(res)
		
		return all_token_ids

	def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
		for line in iterable:
			ids = self.encode(line)
			yield from ids

	def __encode(self, text: str) -> List[int]:
		all_ids = []
		merge_rank: Dict[Tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.__merges)}
		for word in iter_pretokenize(text):
			token_ids = [self.__bytes_to_ids[bytes([id])] for id in word]
			while True:
				# 找到次数最多的pair, 也就是 merge rank 值越小，说明在训练时出现的次数最多
				# 因为出现次数最多的 pair 才会首先被 merge
				rank = 10000000000
				best_pos = -1
				for i in range(len(token_ids) - 1):
					p1 = self.__id_to_bytes[token_ids[i]]
					p2 = self.__id_to_bytes[token_ids[i+1]]
					pair_rank = merge_rank.get((p1, p2), 10000000000)
					if pair_rank < rank:
						rank = pair_rank
						best_pos = i
				# 不能合并
				if best_pos == -1:
					break
				new_bytes = self.__id_to_bytes[token_ids[best_pos]] + self.__id_to_bytes[token_ids[best_pos+1]]
				new_id = self.__bytes_to_ids[new_bytes]
				token_ids[best_pos: best_pos+2] = [new_id]

			all_ids.extend(token_ids)

		return all_ids

	def decode(self, ids: List[int]) -> str:
		'''
		ID 转换为 text
		'''
		all_bytes = b"".join(self.__id_to_bytes[id] for id in ids)
		return all_bytes.decode("utf-8", errors="replace")

	def __group_all_tokens(self, all_word_counter: Counter):
		'''
		根据所有token的计数，构建双向链表，优先级队列
		'''
		pair_postions = defaultdict(set)
		for token_byte_tuple, count in all_word_counter.items():
			if len(token_byte_tuple) < 2:
				continue
			# 将字节值转换为token ID，这里必须转换一下，因为sepcial_token 把前面的tokenid占用了
			token_ids = [self.__bytes_to_ids[bytes([b])] for b in token_byte_tuple]
			pre_node = Node(token_ids[0], count)
			for i in range(1, len(token_ids)):
				node = Node(token_ids[i], count)
				pre_node.next = node
				node.pre = pre_node

				pair = (pre_node.value, node.value)
				pair_postions[pair].add(pre_node)

				pre_node = node
		# 统计所有pair 的计数
		pair_counter = Counter()
		for pair, nodes in pair_postions.items():
			pair_counter[pair] = sum([n.count for n in nodes])

		# 根据统计计数，构建优先级队列
		pq = [SortItem(count, p, (self.__id_to_bytes[p[0]], self.__id_to_bytes[p[1]])) for p, count in pair_counter.items()]
		heapq.heapify(pq)

		return pair_postions, pair_counter, pq

	def __collect_all_word(self, path: str | os.PathLike) -> Counter:
		'''多进程收集所有的word的统计计数'''
		chunks = []
		with open(path, "rb") as f:
			boundaries = find_chunk_boundaries(f, self.__num_chunk, b"<|endoftext|>")
			for start, end in zip(boundaries[:-1], boundaries[1:]):
				f.seek(start)
				chunk = f.read(end - start).decode("utf-8", errors="ignore")
				chunks.append((chunk, self.__special_tokens, self.__special_pattern()))

		all_word_list = Counter()
		with Pool(processes=self.__num_chunk) as pool:
			result_iter = pool.imap_unordered(word_count, chunks)
			for chunk_counter in result_iter:
				all_word_list.update(chunk_counter)

		return all_word_list

	def __get_best_pair(self, pq, pair_counter):
		if not pq:
			return None
		while pq:
			item = heapq.heappop(pq)
			# 延迟删除，因为更新时，只是更新了堆中的计数，但是 pair_counter 是将整个pair都删除了
			if item.id_pair not in pair_counter:
				continue

			if item.count == pair_counter[item.id_pair]:
				return item.id_pair

		return None


	def slow_train(self, path: str | os.PathLike):
		'''
		1. 将字符串转换为bytes对象
		2. 所有的bytes，双层数组存储，外层是bytes对象数组，里层是byte数组
		3. 对里层的byte数组，前后组合，找到次数最多的组合，注意这里组合，是2个bytes（转换id）对象组合（虽然第一次是2个byte）
		但是利于后面的多个byte对象进行两两组合
		4. 将最多的组合分配为新的id，同时更新词表，更新全部组合
		5. 持续迭代直到词表大小达标
		'''
		assert self.__vocab_size >= len(self.__vocab)
		with open(path, "r", encoding = "utf-8") as f:
			text = f.read()

		# 特殊token 进行切分，因为这是单独的token
		sepcial_pattern = self.__special_pattern()
		if sepcial_pattern:
			text_parts = sepcial_pattern.split(text)
		else:
			text_parts = [text]

		token_id_group: list[list[int]] = []
		for t in text_parts:
			# 这里必须排除 special
			if t is None or t in self.__special_tokens:
				continue

			word_bytes = pretokenize(t)
			# 将每一个byte转换为bytes，再转换为id，存储
			for word in word_bytes:
				# 合并时，这样永远是2个不同的id在合并，不然一直是2个bytes对象合并，虽然意义一样，但是效率非常低
				token_id_group.append([self.__bytes_to_ids[bytes([w])] for w in word])

		merge_nums = self.__vocab_size - len(self.__vocab)
		for i in range(merge_nums):
			pair_count = self.__get_pair(token_id_group)
			if not pair_count:
				break

			# 找的次数最多那对, 比较规则，先比较次数，再比较大小(bytes大小)
			best_pair = max(pair_count, key=lambda p:(pair_count[p], self.__id_to_bytes[p[0]],
				self.__id_to_bytes[p[1]]))

			# 分配新id
			new_token_id = len(self.__bytes_to_ids)
			new_bytes = self.__id_to_bytes[best_pair[0]] + self.__id_to_bytes[best_pair[1]]
			self.__bytes_to_ids[new_bytes] = new_token_id
			self.__id_to_bytes[new_token_id] = new_bytes
			self.__merges.append((self.__id_to_bytes[best_pair[0]], self.__id_to_bytes[best_pair[1]]))

			# 更新集合
			token_id_group = self.__merge_pair(token_id_group, best_pair, new_token_id)

		# 更新词表
		self.__vocab = self.__id_to_bytes.copy()



	def __get_pair(self, token_id_group: list[list[int]]):
		pair_count = {}
		for group in token_id_group:
			for pair in zip(group, group[1:]):
				pair_count[pair] = pair_count.get(pair, 0) + 1

		return pair_count

	def __merge_pair(self, token_id_group: list[list[int]], pair: tuple[int, int], new_id: int):
		new_ids_group = []
		for group in token_id_group:
			new_group = []
			i = 0
			while i < len(group):
				if i < len(group) - 1 and (group[i], group[i+1]) == pair:
					new_group.append(new_id)
					i += 2
				else:
					new_group.append(group[i])
					i += 1
			new_ids_group.append(new_group)

		return new_ids_group

	def __special_pattern(self) -> re.Pattern:
		pattern_complied = None
		if self.__special_tokens:
			# 长度从大到小排序，避免切割时，长的 special_token 被小 special_token 切割
			special_token_sort = sorted([t.encode("utf-8") for t in self.__special_tokens], key = len, reverse = True)
			escape_tokens = [re.escape(s.decode("utf-8")) for s in special_token_sort]
			pattern_complied = re.compile(f"({'|'.join(escape_tokens)})")

		return pattern_complied

	@property
	def vocab(self):
		"""返回词表字典，从token ID到bytes的映射"""
		return self.__vocab

	@property
	def merges(self):
		"""返回BPE合并列表，每个元素是(bytes, bytes)的元组"""
		return self.__merges

	@property
	def vocab_size(self):
		"""返回词表大小"""
		return self.__vocab_size

	@property
	def special_tokens(self):
		"""返回特殊token列表"""
		return self.__special_tokens

	def _set_vocab_data(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]):
		"""设置词汇表数据的内部方法"""
		self.__bytes_to_ids = {v: k for k, v in vocab.items()}
		self.__id_to_bytes = vocab
		self.__merges = merges
		self.__vocab = vocab

	@classmethod
	def new_instance(
		cls,
		vocab: Dict[int, bytes],
		merges: List[Tuple[bytes, bytes]],
		special_tokens: List[str],
	):
		instance = cls(vocab_size=len(vocab), num_chunk=4, special_tokens=special_tokens)
		instance._set_vocab_data(vocab, merges)
		return instance

