# BERT_with_Residual_vs_Highway
Comparing between residual stream and highway stream in transformers(BERT) .

### Codabase Descriptions:
* **Scripts**: Where you will find model skeleton used, prepare dataset, training script, config file of hyperparameters.
* **Assets folder**: Where you will find the results of trainig(train_state ,learning_curve).

### Compute:
* Google Collab(Tesla K80)

### Dataset:
* Dataset: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt


### Architecture:


![alt text](https://github.com/mhmdsabry/BERT_with_Residual_vs_Highway/blob/main/model_architecture/BERT%20Residual_vs_Highway.drawio.png)
### Model Config:

* num encoders:8
* num attention heads:8
* hidden dim: 512
* max tokens: 128
* AdamW optimizer

### Results

Results for the current hyperparameter configs and model config(all can be found in config.ini, follow commits for different experiments config)
The following results are preliminary, just to show basic differences, also check the learning curves in the assets folder. 

#### Residual BERT
* num params: 2.535328e+07
* Training time: 9847396.0ms
* best loss: 4.2794

#### Highway BERT
* num params: 2.955578e+07
* Training time: 10547382.0ms
* best loss: 4.3240

### Things to be Done

I don't have the proper computing to explore this flexibly. So I hope the community makes a trial on this with the following ideas:
* enlarge the model size
* better and more datasets, here I used a toy dataset(modeling chars), just to make sure the code works and has some basic results.
* best tune each of the networks, just to see what's each requirement to give its best performance.
* Because it's for fun maybe doing a where-when-what analysis in both networks will be great!



### References:
* Training Very Deep Networks(https://arxiv.org/abs/1507.06228)
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (https://arxiv.org/abs/1810.04805)
* HIGHWAY AND RESIDUAL NETWORKS LEARN UNROLLED ITERATIVE ESTIMATION (https://arxiv.org/abs/1612.07771)
* Residual Networks Behave Like Ensembles of Relatively Shallow Networks (https://arxiv.org/abs/1605.06431)

**Note:** If anything is not clear and you need to discuss some stuff or have any suggestions, or notes, reach out at mhmd.sabry.ab@gmail.com 
