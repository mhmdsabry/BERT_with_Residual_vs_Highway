import configparser
import argparse

import time

import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
						datefmt="%m/%d/%Y %H:%M:%S",
						level=logging.INFO)

import torch
from torch.utils.data import TensorDataset

from bert_model import bertModel, bertConfig
from trainer import Trainer, TrainerConfig
from utils import *
from prepare_dataset import *

logger = logging.getLogger(__name__)
SEED = 1738
seed_everything(SEED)

#command line parser for config file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("-c","--config",dest="filename",help="Pass config file",metavar="FILE")
args = parser.parse_args()
config.read(args.filename)

#hyperparameters:
#dataset
block_size = int(config['model_config']['block_size'])

dataset_path = config['dataset']['dataset_path']
train_len = int(config['dataset']['train_len'])
eval_len = int(config['dataset']['eval_len'])
text = open('input.txt', 'r').read()
train_dataset = CharDataset(text[:train_len], block_size)
eval_dataset = CharDataset(text[train_len:train_len+eval_len], block_size)

#model
hidden_dimension = int(config['model_config']['hidden_dimension'])
num_encoders = int(config['model_config']['num_encoders'])
depth_enabler = config['model_config']['depth_enabler']
attention_head = int(config['model_config']['attention_head'])

#training
max_epoch = int(config['training_config']['max_epoch'])
train_batch_size = int(config['training_config']['train_batch_size'])
eval_batch_size = int(config['training_config']['eval_batch_size'])
num_workers = int(config['training_config']['num_workers'])
learning_rate = float(config['training_config']['learning_rate'])
warmup_steps = float(config['training_config']['warmup_steps'])
weight_decay = float(config['training_config']['weight_decay'])
betas_1 = float(config['training_config']['betas_1'])
betas_2 = float(config['training_config']['betas_2'])
ckpt_path = config['training_config']['ckpt_path']
learning_curve_path = config['training_config']['learning_curve_path']
vocab_size = train_dataset.get_vocab_size() if train_dataset.get_vocab_size() > eval_dataset.get_vocab_size() else eval_dataset.get_vocab_size() 
#set model
model_config = bertConfig(

							n_embed = hidden_dimension,
							depth_enabler = depth_enabler,
							n_head = attention_head,
							n_encoders = num_encoders,
							vocab_size = vocab_size,
							tokens_size = block_size,)

model = bertModel(model_config)

#prepare trainer
training_config = TrainerConfig(
									max_epoch = max_epoch,
									train_batch_size = train_batch_size,
									eval_batch_size = eval_batch_size,
									num_workers = num_workers,
									learning_rate = learning_rate,
									warmup_steps = warmup_steps,
									weight_decay = weight_decay,
									betas_1 = betas_1,
									betas_2 = betas_2,
									ckpt_path = ckpt_path)


if __name__ == "__main__":
	start = time.time()
	
	trainer = Trainer(model, train_dataset, eval_dataset, training_config)
	trainer.train()
	
	elapsed_time = time.time() - start
	logger.info(f"Training time:{elapsed_time/60}m")

	if max_epoch == 1:
		plot_learning_curve_iter(f'{ckpt_path}_{max_epoch}epoch_train_state.json',
						 	f'{learning_curve_path}_{max_epoch}iter_{depth_enabler}')

	if max_epoch > 1:
		plot_learning_curve_epoch(f'{ckpt_path}_{max_epoch}epoch_train_state.json',
						 	f'{learning_curve_path}_{max_epoch}epoch_{depth_enabler}')




