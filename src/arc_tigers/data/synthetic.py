import numpy as np
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


def get_synthetic_data(
    n_samples: int,
    imbalance: float,
    seed: int | None = None,
    add_text: bool = True,
    tokenizer: PreTrainedTokenizer | None = None,
) -> Dataset:
    """
    Generate a synthetic binary classification dataset with a specified imbalance ratio.

    Args:
        n_samples: Number of samples in the dataset.
        imbalance: Imbalance ratio for the positive class (0 < imbalance < 1).
        seed: Random seed for reproducibility.
        add_text: Whether to add a dummy text column (containing 'yes' if label is 1,
            'no' if label is 0))

    Returns:
        Dataset containing synthetic 'label' and optional 'text' features.
    """
    rng = np.random.default_rng(seed)
    data_dict = {"label": (rng.random(n_samples) < imbalance).astype(int)}
    if add_text:
        data_dict["text"] = [
            "yes" if label == 1 else "no" for label in data_dict["label"]
        ]

    dataset = Dataset.from_dict(data_dict)
    if add_text and tokenizer:
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True), batched=True
        )
    return dataset
