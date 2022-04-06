import math
import logging

import copy

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class CharDataset(Dataset):
	def __init__(self, data, block_size):
		chars = sorted(list(set(data)))
		data_size, vocab_size = len(data), len(chars)
		logger.info('data has %d characters, %d unique.'%(data_size, vocab_size))

		self.tokenizer = {ch:i for i, ch in enumerate(chars)}
		self.decoder = {i:ch for i,ch in enumerate(chars)}

		#Add MASK token
		self.tokenizer['MASK'] = 103
		self.decoder[103] = 'MASK'

		self.block_size = block_size
		self.vocab_size = vocab_size
		self.data = data

	def get_vocab_size(self):
		return self.vocab_size

	def mask(self,input_ids):
		labels = copy.deepcopy(input_ids)
		rand = torch.rand(input_ids.shape)
		mask_arr = rand < 0.15

		x = input_ids.masked_fill(mask_arr==True, 103)
		y = labels.masked_fill(mask_arr==False, -100)

		return x, y

	def __len__(self):
		return len(self.data) - self.block_size

	def __getitem__(self,idx):
		chunk = self.data[idx:idx+self.block_size]

		encoding = [self.tokenizer[c] for c in chunk]
		input_ids, labels = self.mask(torch.tensor(encoding, dtype=torch.long))

		return input_ids, labels #x,y







