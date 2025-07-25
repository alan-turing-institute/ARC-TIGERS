from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

EXPECTED_KEYS = {"text", "label", "len"}


def hf_train_test_split(dataset: Dataset, **split_kwargs) -> tuple[Dataset, Dataset]:
    """Splits a dataset into train and test sets, returning them as a tuple rather than
    a dict like the default HF datasets `train_test_split` method."""
    split = dataset.train_test_split(**split_kwargs)
    return split["train"], split["test"]


def tokenize_data(data: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Loads and preprocesses the Reddit dataset based on the specified configuration.

    Returns:
        tuple: A tuple containing the training dataset, evaluation dataset, and test
        dataset.
    """
    return data.map(
        lambda batch: tokenizer(batch["text"], padding=True, truncation=True),
        batched=True,
    )
