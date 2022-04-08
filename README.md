# BERT_with_Residual_vs_Highway
Comparing between residual stream and highway stream in transformers(BERT) .

### Codabase Descriptions:
* **Scripts**: Where you will find model skeleton used, prepare dataset, training script, config file of hyperparameters.
* **Assets folder**: Where you will find the results of trainig(train_state ,learning_curve).

### Compute:
* Google Collab(Tesla K80)

### Dataset:
* Dataset: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

### Model Config:
* num encoders:8
* num attention heads:8
* hidden dim: 512
* max tokens: 128
* AdamW optimizer


### Architecture:


![alt text](https://github.com/mhmdsabry/BERT_with_Residual_vs_Highway/blob/main/model_architecture/BERT%20Residual_vs_Highway.drawio.png)

### Results
#### Residual BERT:
* num params: 2.535328e+07
* training time:
* best loss:

#### Highway BERT:
* num params: 2.955578e+07
* training time:
* best loss:

### Concluding Remarks

### Future works

### References:
* Training Very Deep Networks(https://arxiv.org/abs/1507.06228)
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (https://arxiv.org/abs/1810.04805)
* HIGHWAY AND RESIDUAL NETWORKS LEARN UNROLLED ITERATIVE ESTIMATION (https://arxiv.org/abs/1612.07771)
* Residual Networks Behave Like Ensembles of Relatively Shallow Networks (https://arxiv.org/abs/1605.06431)
