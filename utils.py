import math 
import json

import random, os
import numpy as np 
import torch

import transformers

import matplotlib.pyplot as plt


def seed_everything(seed: int):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True

def plot_learning_curve(state, saving_path=None):
	state = json.load(open(state))

	train_loss = state.get('train_loss')
	val_loss = state.get('eval_loss')
	epoch = state.get('epoch')

	fig = plt.figure()
	ax = plt.axes()
	ax.plot(epoch, train_loss,label='Train Loss')
	ax.plot(epoch, val_loss, label='Eval Loss')
	plt.xlabel("Epoch")
	plt.ylabel("Loss");
	plt.legend()

	if saving_path is not None:
		plt.savefig(f'{saving_path}.png')

	plt.show()
