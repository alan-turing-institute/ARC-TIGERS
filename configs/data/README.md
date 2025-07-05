## `configs/data`

Here the config `.yaml` files should have two arguments: `data_name` and `data_args`.

These should refer to the dataset to load from `data` and the arguments to be passed to the corresponding function which loads this dataset for training and evaluation.

eg:
```
data_name: reddit_dataset_12
data_args:
  n_rows: 15000000
  task: multi-class
  target_config: sport
  balanced: true
```

Random seeds along with training hyperparameters should be passed in the experiment config from `configs/experiment`.
