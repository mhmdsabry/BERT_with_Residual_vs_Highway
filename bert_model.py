import math
import logging

import numpy as np 

import torch
import torch.nn as nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)

class bertConfig:
	pdrop_attn = 0.1
	pdrop_resid = 0.1
	embd_pdrop = 0.1
	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)

class selfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embed % config.n_head == 0 

		self.keys = nn.Linear(config.n_embed, config.n_embed)
		self.querys = nn.Linear(config.n_embed, config.n_embed)
		self.values = nn.Linear(config.n_embed, config.n_embed)

		self.attn_drop = nn.Dropout(config.pdrop_attn)
		self.resid_drop = nn.Dropout(config.pdrop_resid)

		self.proj = nn.Linear(config.n_embed, config.n_embed)
		self.n_head = config.n_head

	def forward(self, x):
		B, T, C = x.size()

		keys = self.keys(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2) #(B,nh,T,h)
		querys = self.querys(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)#(B,nh,T,h)
		values = self.values(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)#(B,nh,T,h)

		attn = (querys @ keys.transpose(-1,-2)) * (1 / math.sqrt(keys.size(-1))) #(B,nh,T,h)@(B,nh,h,T)->(B,nh,T,T)
		attn = F.softmax(attn, dim=-1)
		attn = self.attn_drop(attn)
		y = attn @ values #(B,nh,T,T) @ (B,nh,T,h) -> (B,nh,T,h)
		y = y.transpose(1,2).contiguous().view(B, T, C)

		y = self.proj(y)

		return y

class block(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.attn_ln = nn.LayerNorm(config.n_embed)
		self.attn_block = selfAttention(config)

		self.mlp_ln = nn.LayerNorm(config.n_embed)
		self.MLP_block = nn.Sequential(
										nn.Linear(config.n_embed, 4*config.n_embed),
										nn.GELU(),
										nn.Linear(4*config.n_embed, config.n_embed),
										nn.Dropout(config.pdrop_resid)
										)

		if config.depth_enabler =="Highway":
			self.transform_gate_attn = nn.Sequential(
												nn.Linear(config.n_embed, config.n_embed),
												nn.Sigmoid())

			self.transform_gate_mlp = nn.Sequential(
												nn.Linear(config.n_embed, config.n_embed),
												nn.Sigmoid())

		self.config = config

	def forward(self, x):
		if self.config.depth_enabler=="Residual" or "default":
			x = x + self.attn_block(self.attn_ln(x))
			x = x + self.MLP_block(self.mlp_ln(x))
		elif self.config.depth_enabler=="Highway":
			x = x * (1-self.transform_gate_attn(x)) + self.attn_block(self.attn_ln(x)) * self.transform_gate_attn(x)
			x = x * (1-self.transform_gate_mlp(x)) + self.MLP_block(self.mlp_ln(x)) * self.transform_gate_mlp(x)

		return x 


class bertModel(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.embed  = nn.Embedding(config.vocab_size, config.n_embed)
		self.pos_embed = nn.Parameter(torch.ones(1, config.tokens_size, config.n_embed))
		self.embed_drop = nn.Dropout(config.embd_pdrop)

		self.encoders = nn.Sequential(*[block(config) for _ in range(config.n_encoders)])
	
		self.head_ln = nn.LayerNorm(config.n_embed)
		self.head = nn.Linear(config.n_embed, config.vocab_size)
	
		self.apply(self._init_weight)
		self.config =config
		logger.info("Number of parameters: %e",sum(p.numel() for p in self.parameters()))

	def _init_weight(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
			if isinstance(module, nn.Linear) and module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.LayerNorm):
			module.weight.data.fill_(1.0)
			module.bias.data.zero_()

	def bert_optimizer(self, train_config):

		decay = set()
		no_decay = set()
		whitelist_weight_modules = (torch.nn.Linear, )
		blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
		for mn, m in self.named_modules():
			for pn, p in m.named_parameters():
				fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

				if pn.endswith('bias'):
					# all biases will not be decayed. time-weight will not be decayed.
					no_decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
					# weights of whitelist modules will be weight decayedf
					decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
					# weights of blacklist modules will NOT be weight decayed
					no_decay.add(fpn)

		# special case the position embedding parameter in the root GPT module as not decayed
		no_decay.add('pos_embed')

		# validate that we considered every parameter
		param_dict = {pn: p for pn, p in self.named_parameters()}
		inter_params = decay & no_decay
		union_params = decay | no_decay
		assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
		assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
													% (str(param_dict.keys() - union_params), )

		# create the pytorch optimizer object
		optim_groups = [
			{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
			{"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
		]

		
		optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=(train_config.betas_1,train_config.betas_2))
		return optimizer




	def forward(self, x,target=None):
		B, T = x.size()
		assert T<=self.config.tokens_size, "Tokens size is exhausted!"

		embeddings = self.embed(x)
		positions = self.pos_embed[:,:T,:]
		x = self.embed_drop(embeddings + positions)

		x = self.encoders(x)

		logits = self.head(self.head_ln(x))

		loss=None
		if target is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

		return logits, loss  






















