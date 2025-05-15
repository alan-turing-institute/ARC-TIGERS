## `configs/experiment`

Here the config `.yaml` files should have five fields:

- `model_config`: the name of the model config being used inside `configs/model`
- `data_config`: the name of the data config being used inside `configs/data`
- `exp_name`: the name of the experiment when it is saved
- `random_seed`: the random seed to use throughout the training and evaluation
- `train_kwargs`: the training arguments such as epochs


For example:
```
model_config: distilbert
data_config: reddit_football
exp_name: football_distilbert
random_seed: 42
train_kwargs:
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  warmup_steps: 500
  weight_decay: 0.01
  save_total_limit: 3
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end : True
```
