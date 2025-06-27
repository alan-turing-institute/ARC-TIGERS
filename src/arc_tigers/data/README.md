# `get_reddit_data.py`

This module provides the main function for loading and preprocessing Reddit datasets for classification experiments. It supports multi-class, one-vs-all, and data-drift settings, with options for balancing, imbalancing.

## `get_reddit_data`:

Description: Loads and preprocesses the Reddit dataset based on the specified configuration. Handles different experimental settings (multi-class, one-vs-all, data-drift), applies tokenization, balancing or imbalancing, and returns train, eval, and test splits along with metadata.

### Arguments:

- `setting` (str):
  - The classification setting. Options: "multi-class", "one-vs-all", "data-drift".
- `target_config` (str):
  - The target configuration to use (e.g., "football", "news", etc.).
- `balanced` (bool):
  - Whether to balance the dataset by resampling classes.
- `n_rows` (int):
  - Number of rows to load from the dataset.
- `tokenizer` (transformers.PreTrainedTokenizer or None):
  - Tokenizer to use for preprocessing.
- `class_balance` (float or None):
  - If set, imbalances the dataset to the specified class ratio.
- `all_classes` (bool):
  - If True (multi-class only), uses all classes for training.

### Returns:

A tuple: (`train_data`, `eval_data`, `test_data`, `meta_data`)

- `train_data`:
  - Training split (tokenized and preprocessed)
- `eval_data`:
  - Evaluation split (tokenized and preprocessed)
- `test_data`:
  - Test split (tokenized and preprocessed)
- `meta_data`:
  - Dictionary with label mappings and class counts

### How it works:

Loads the appropriate data split based on the setting and target_config.
Applies class mapping and filtering according to the experiment type.
Tokenizes the data using the provided tokenizer.
Optionally balances or imbalances the dataset.
Splits the training data into train/eval splits.
Returns the processed datasets and metadata.

### Example usage:

For example, loading the one-vs-all setting of the football split, generated using 10,000,000 rows from the original reddit dataset:

```python
from transformers import AutoTokenizer
from arc_tigers.data.reddit_data import get_reddit_data

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_data, eval_data, test_data, meta_data = get_reddit_data(
        setting="one-vs-all",
        target_config="football",
        balanced=True,
        n_rows=10000000,
        tokenizer=tokenizer,
        )
```

At this point the tokenized dataset is loaded, with classes balanced according to experiment requirements and tokenized ready for a dataloader (if a tokenizer is passed).

### Notes:

- The function expects preprocessed data splits to exist in the expected directory structure.
- Available target_config options are in the dictionaries at the top of reddit_data.py.
- For more details on balancing/imbalancing, see arc_tigers/data/utils.py.

# Note on class balancing

The `class_balance` parameter in get_reddit_data controls whether and how the dataset is artificially imbalanced for binary classification experiments. Imbalancing is performed by [`imbalance_binary_dataset`](https://github.com/alan-turing-institute/ARC-TIGERS/blob/30-one-vs-all-classification/src/arc_tigers/data/utils.py#L35) in `arc_tigers/data/utils.py`.

1. When `balanced=True`
   - The dataset is balanced using the `balance_dataset` function.
   - This means the number of samples for each class is made equal by downsampling the majority class.
   - The `class_balance` parameter is ignored in this case.

2. When `balanced=False` and `class_balance=None`
   - The dataset is left in its natural state.
   - No artificial balancing or imbalancing is performed.
   - The class distribution will reflect the original data.

3. When `balanced=False` and `class_balance` is set (e.g., `class_balance=0.1` or c`lass_balance=-0.1`)
    - The dataset is imbalanced using the `imbalance_binary_dataset` function.
    - The `class_balance` parameter must be a float in the range -1.0 to 1.0 (exclusive of zero).
    - The sign of `class_balance` determines which class is the minority:
        - Positive value (e.g., 0.1):
          - Class 0 is the majority, class 1 is the minority.
          - The number of class 1 samples will be `int(n_class_0 * class_balance`).
        - Negative value (e.g., -0.1):
          - Class 1 is the majority, class 0 is the minority.
          - The number of class 0 samples will be `int(n_class_1 * abs(class_balance))`.
    - If there are not enough samples in the minority class to achieve the specified ratio, the function will:
        - Downsample the majority class to match the available minority samples.
        - The function randomly samples the required number of indices from each class (without replacement) to create the imbalanced dataset.

   - Example 1:
      - If class_balance=0.2 and you have 1000 class 0 and 800 class 1:
        - Class 0 (majority): 1000 samples
        - Class 1 (minority): int(1000 * 0.2) = 200 samples (randomly sampled from class 1)
        -    Final dataset: 1000 class 0 + 200 class 1
   - Example 2:
        - If class_balance=-0.1 and you have 1000 class 0 and 800 class 1:
        - Class 1 (majority): 800 samples
        - Class 0 (minority): int(800 * 0.1) = 80 samples (randomly sampled from class 0)
        - Final dataset: 80 class 0 + 800 class 1

4. Effect on Train, Eval, and Test Splits
    - The balancing or imbalancing is applied separately to the training, evaluation, and test splits.
    - This ensures that each split has the desired class distribution according to the `class_balance` parameter.

5. Typical Use Cases
    - Use `balanced=True` for fair comparison between classes.
   - Use `class_balance` to simulate rare event detection or to test model robustness to class imbalance.
    - Leave both unset to use the dataset as-is.
