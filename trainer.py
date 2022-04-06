import math
import json

import logging

import numpy as np 

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class TrainerConfig:
	def __init__(self,**kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)


class Trainer(TrainerConfig):
	def __init__(self, model, trainset, evalset, train_config):
		self.model = model
		self.trainset = trainset
		self.evalset = evalset
		self.config = train_config

		self.device='cpu'
		if torch.cuda.is_available():
			self.device = torch.cuda.current_device()
			self.model = self.model.to(self.device)


	def save_checkpoints(self, timeline):
		model = self.model
		logger.info("Saving at %s", self.config.ckpt_path)
		torch.save(model.state_dict(), f"{self.config.ckpt_path}{timeline}")


	def train(self):
		train_state = {
		"epoch":[],
		"train_loss":[],
		"eval_loss":[],
		"best_loss_epoch":set()
		}
		config = self.config
		model = self.model
		optimizer = model.bert_optimizer(config)

		training_steps = int(len(self.trainset) / config.train_batch_size * config.max_epoch)

		scheduler = get_linear_schedule_with_warmup(
					optimizer,
					num_warmup_steps=config.warmup_steps*training_steps,
					num_training_steps=training_steps)

		def train_loop_fn(train_dataloader):
			losses = []
			model.train()

			for itr, (x,y) in enumerate(train_dataloader):
				x = x.to(self.device)
				y = y.to(self.device)

				optimizer.zero_grad()
				pred, loss = self.model(x, y)

				losses.append(loss.item())

				if itr % 10 and itr>0:
					logger.info(f"Itr:{itr}, loss:{loss}")

				loss.backward()
				optimizer.step()
				scheduler.step()

			return float(np.mean(losses))

		def eval_loop_fn(eval_dataloader):
			losses = []
			model.eval()

			for itr, (x,y) in enumerate(eval_dataloader):
				x = x.to(self.device)
				y = y.to(self.device)

				pred, loss = self.model(x,y)
				losses.append(loss.item())

			return float(np.mean(losses))


		train_dataloader = DataLoader(
			self.trainset,
			batch_size = config.train_batch_size,
			num_workers = config.num_workers,
			drop_last=True)
		eval_dataloader = DataLoader(
			self.evalset,
			batch_size = config.eval_batch_size,
			num_workers=config.num_workers,
			drop_last=True)


		best_loss = float('inf')
		for epoch in range(config.max_epoch):
			logger.info(f'==========Epoch:{epoch+1}/{config.max_epoch}==========')
			train_state['epoch'].append(epoch+1)

			train_loss = train_loop_fn(train_dataloader)
			train_state['train_loss'].append(train_loss)

			eval_loss = eval_loop_fn(eval_dataloader)
			train_state['eval_loss'].append(eval_loss)

			goodModel = eval_loss < best_loss

			if config.ckpt_path is not None and goodModel:
				best_loss = eval_loss
				train_state['best_loss_epoch']=(best_loss, epoch+1)
				self.save_checkpoints(f"_{config.max_epoch}epoch_best_model")

		with open(f'{config.ckpt_path}_{config.max_epoch}epoch_train_state.json',"w") as fp:
			json.dump(train_state, fp)

		self.save_checkpoints(f"_{config.max_epoch}epoch_last_model")





